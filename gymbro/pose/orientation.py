"""Gravity alignment for cameras that aren't level.

MediaPipe's `pose_world_landmarks` are metric, but they are expressed in a
*camera-aligned* frame: +Y is up in the image, not up in the world. Put the
laptop on a table and train underneath it and the whole skeleton arrives
rotated by the camera's pitch. Anything measured against Y then reads the
camera's tilt rather than the body's posture -- a 60-degree downward camera
shifts a trunk-lean measurement by a full 60 degrees.

This module estimates which way is actually up, in camera coordinates, and
rotates poses into a gravity-aligned frame before anything measures them. Do it
here, once, at the source, rather than patching each rule.

Two ways to calibrate, because someone training in a cramped space may not be
able to stand up in frame:

  standing  - hold still upright for a moment; hip->shoulder is up.
  floor     - hold any position with both feet planted; the floor plane's
              normal is up. Works lying down, which is most glute work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .skeleton import Pose

# Below this, the calibration is too uncertain to rely on.
MIN_CONFIDENCE = 0.55


@dataclass
class Orientation:
    """A rotation from camera-aligned world coordinates into gravity-aligned.

    `up` is the unit vector, in camera coordinates, that points against gravity.
    """

    up: np.ndarray
    confidence: float
    method: str
    samples: int = 0

    @property
    def tilt_degrees(self) -> float:
        """How far the camera is from level. 0 = level, 90 = straight down."""
        # A level camera sees gravity as (0, 1, 0).
        return float(np.degrees(np.arccos(np.clip(self.up[1], -1.0, 1.0))))

    @property
    def reliable(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE

    def rotation(self) -> np.ndarray:
        """3x3 matrix mapping camera-aligned vectors to gravity-aligned ones."""
        return _rotation_between(self.up, np.array([0.0, 1.0, 0.0]))

    def apply(self, pose: Pose) -> Pose:
        """Return a copy of `pose` with world landmarks gravity-aligned."""
        if not self.reliable:
            return pose
        rotated = pose.world @ self.rotation().T
        return Pose(
            world=rotated, pixels=pose.pixels, confidence=pose.confidence,
            timestamp=pose.timestamp, source=pose.source,
            meta={**pose.meta, "gravity_aligned": True, "camera_tilt": self.tilt_degrees},
        )

    def to_dict(self) -> dict:
        return {
            "up": [float(x) for x in self.up],
            "confidence": self.confidence,
            "method": self.method,
            "samples": self.samples,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Orientation":
        return cls(
            up=np.asarray(d["up"], dtype=np.float64), confidence=d["confidence"],
            method=d["method"], samples=d.get("samples", 0),
        )


def identity() -> Orientation:
    """A level camera: no correction. Used when calibration is unavailable."""
    return Orientation(up=np.array([0.0, 1.0, 0.0]), confidence=1.0, method="assumed_level")


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector `a` onto unit vector `b`."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return np.eye(3)
    a, b = a / na, b / nb

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-9:
        # Parallel or antiparallel.
        if c > 0:
            return np.eye(3)
        # 180 degrees: rotate about any axis perpendicular to a.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, axis)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * K @ K

    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))


# ----------------------------------------------------------------- calibration


class OrientationCalibrator:
    """Accumulates frames and estimates which way is up.

    Both methods need the subject to hold still; `stability` reports whether
    they have, so the caller can tell the user to stop moving rather than
    silently producing a bad calibration.
    """

    def __init__(self, method: str = "standing", min_samples: int = 20):
        if method not in ("standing", "floor"):
            raise ValueError(f"unknown calibration method: {method!r}")
        self.method = method
        self.min_samples = min_samples
        self._vectors: list[np.ndarray] = []

    def add(self, pose: Pose) -> bool:
        """Feed a frame. Returns True if it was usable."""
        vector = (
            self._standing_up(pose) if self.method == "standing"
            else self._floor_up(pose)
        )
        if vector is None:
            return False
        self._vectors.append(vector)
        return True

    @staticmethod
    def _standing_up(pose: Pose) -> np.ndarray | None:
        """Up is hip midpoint -> shoulder midpoint, while standing upright."""
        needed = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if not pose.visible(needed, threshold=0.6):
            return None
        shoulder = (pose.joint("left_shoulder") + pose.joint("right_shoulder")) / 2
        hip = (pose.joint("left_hip") + pose.joint("right_hip")) / 2
        v = shoulder - hip
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else None

    @staticmethod
    def _floor_up(pose: Pose) -> np.ndarray | None:
        """Up is the normal of the floor plane, for someone lying supine.

        The floor contacts are the two shoulders and the heels -- all three
        stay on the mat whether you are resting or mid-bridge. An earlier
        version used the heels and the *hips*, which fails precisely because
        the hips are what lift during the movement being measured: the plane
        pivoted with every rep.

        Shoulders plus heel midpoint also make a wide, well-conditioned
        triangle, where heels-plus-hips is nearly collinear when lying flat.
        """
        needed = ("left_shoulder", "right_shoulder", "left_heel", "right_heel")
        if not pose.visible(needed, threshold=0.5):
            return None

        left_sh = pose.joint("left_shoulder")
        right_sh = pose.joint("right_shoulder")
        heel_mid = (pose.joint("left_heel") + pose.joint("right_heel")) / 2

        normal = np.cross(right_sh - left_sh, heel_mid - left_sh)
        n = np.linalg.norm(normal)
        if n < 1e-6:
            return None
        normal = normal / n

        # Sign: up points away from the floor, i.e. toward the raised hips.
        # Body thickness alone puts the hips above the shoulder-heel plane.
        if not pose.visible(("left_hip", "right_hip"), threshold=0.5):
            return None
        hip = (pose.joint("left_hip") + pose.joint("right_hip")) / 2
        if np.dot(normal, hip - left_sh) < 0:
            normal = -normal
        return normal

    @property
    def stability(self) -> float:
        """How consistent the samples are, 0-1. Low means they were moving."""
        if len(self._vectors) < 2:
            return 0.0
        stacked = np.stack(self._vectors)
        mean = stacked.mean(axis=0)
        n = np.linalg.norm(mean)
        if n < 1e-9:
            return 0.0
        mean = mean / n
        # Mean resultant length of the unit vectors: 1.0 = perfectly still.
        return float(np.clip(np.mean(stacked @ mean), 0.0, 1.0))

    @property
    def ready(self) -> bool:
        return len(self._vectors) >= self.min_samples

    def result(self) -> Orientation | None:
        if not self.ready:
            return None
        stacked = np.stack(self._vectors)
        # Median is robust to the odd bad frame; renormalize after.
        up = np.median(stacked, axis=0)
        n = np.linalg.norm(up)
        if n < 1e-9:
            return None
        up = up / n

        return Orientation(
            up=up, confidence=self.stability, method=self.method,
            samples=len(self._vectors),
        )

    def reset(self) -> None:
        self._vectors.clear()


def describe_tilt(orientation: Orientation) -> str:
    """Plain-language summary, for telling the user what the camera is doing."""
    tilt = orientation.tilt_degrees
    if tilt < 12:
        return f"Camera is roughly level ({tilt:.0f} deg)."
    if tilt < 35:
        return f"Camera is tilted {tilt:.0f} deg - fine, correcting for it."
    if tilt < 65:
        return (f"Camera is steeply angled ({tilt:.0f} deg). Correcting, but "
                f"depth measurements get less reliable past about 45 deg.")
    return (f"Camera is nearly overhead ({tilt:.0f} deg). Corrections applied, "
            f"but expect reduced accuracy on anything depth-dependent.")
