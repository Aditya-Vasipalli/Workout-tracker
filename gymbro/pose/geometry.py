"""Geometry for pose analysis.

Two things here that the old tracker got wrong:

1. `letterbox` preserves aspect ratio. The old code resized 640x480 straight to
   256x256, compressing x by 0.400 and y by 0.533. Angles measured in that
   warped space are systematically wrong -- a true 45 degree limb reads as 53.

2. Angles are computed in 3D on metric world landmarks, not on projected pixels.
   A 2D angle is only correct when the limb plane is parallel to the sensor;
   any rotation collapses it toward the camera plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------- letterboxing


@dataclass(frozen=True)
class LetterboxTransform:
    """Maps between original-image pixels and letterboxed-model-input pixels."""

    scale: float
    pad_x: int
    pad_y: int
    out_size: int

    def to_original(self, xy: np.ndarray) -> np.ndarray:
        """Map model-input coords (pixels in the padded square) back to source."""
        xy = np.asarray(xy, dtype=np.float64)
        out = np.empty_like(xy)
        out[..., 0] = (xy[..., 0] - self.pad_x) / self.scale
        out[..., 1] = (xy[..., 1] - self.pad_y) / self.scale
        return out

    def normalized_to_original(
        self, xy_norm: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """Map [0,1] coords relative to the padded square back to source pixels.

        Models like MoveNet emit normalized coordinates over their square input.
        Feeding those straight into angle math without undoing the padding is
        what produced the distortion in the original tracker.
        """
        xy_norm = np.asarray(xy_norm, dtype=np.float64)
        px = xy_norm * self.out_size
        return self.to_original(px)


def letterbox_params(width: int, height: int, out_size: int) -> LetterboxTransform:
    """Compute the aspect-preserving fit of a WxH image into a square."""
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {width}x{height}")
    scale = min(out_size / width, out_size / height)
    new_w, new_h = round(width * scale), round(height * scale)
    return LetterboxTransform(
        scale=scale,
        pad_x=(out_size - new_w) // 2,
        pad_y=(out_size - new_h) // 2,
        out_size=out_size,
    )


def letterbox(image: np.ndarray, out_size: int) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize preserving aspect ratio and pad to a square. Requires cv2."""
    import cv2  # local import: core geometry stays importable without OpenCV

    height, width = image.shape[:2]
    tf = letterbox_params(width, height, out_size)
    new_w, new_h = round(width * tf.scale), round(height * tf.scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((out_size, out_size, image.shape[2]), dtype=image.dtype)
    canvas[tf.pad_y : tf.pad_y + new_h, tf.pad_x : tf.pad_x + new_w] = resized
    return canvas, tf


# ------------------------------------------------------------------- 3D angles


def angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at vertex `b`, in degrees. Points are 3D (or 2D).

    Uses atan2 of the cross/dot magnitudes rather than arccos(dot), which is
    numerically unstable near 0 and 180 degrees -- exactly the lockout and
    full-flexion positions that matter most for rep detection.
    """
    a, b, c = (np.asarray(p, dtype=np.float64) for p in (a, b, c))
    ba, bc = a - b, c - b

    if ba.shape[-1] == 2:  # promote to 3D so cross() yields a vector
        ba = np.append(ba, 0.0)
        bc = np.append(bc, 0.0)

    cross = np.linalg.norm(np.cross(ba, bc))
    dot = float(np.dot(ba, bc))
    if np.linalg.norm(ba) < 1e-9 or np.linalg.norm(bc) < 1e-9:
        return float("nan")
    return float(np.degrees(np.arctan2(cross, dot)))


def signed_angle_about_axis(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, axis: np.ndarray
) -> float:
    """Angle at `b` signed by rotation direction about `axis`, in degrees.

    Needed for asymmetry checks: unsigned angles cannot distinguish a knee
    caving inward from one flaring outward.
    """
    a, b, c, axis = (np.asarray(p, dtype=np.float64) for p in (a, b, c, axis))
    ba, bc = a - b, c - b
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return float("nan")
    axis = axis / n

    # Project into the plane normal to the axis.
    ba -= np.dot(ba, axis) * axis
    bc -= np.dot(bc, axis) * axis
    if np.linalg.norm(ba) < 1e-9 or np.linalg.norm(bc) < 1e-9:
        return float("nan")

    return float(
        np.degrees(np.arctan2(float(np.dot(np.cross(ba, bc), axis)), float(np.dot(ba, bc))))
    )


def point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Perpendicular distance from `p` to the line through `a` and `b`."""
    p, a, b = (np.asarray(x, dtype=np.float64) for x in (p, a, b))
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-9:
        return float(np.linalg.norm(p - a))
    return float(np.linalg.norm(np.cross(ab, p - a)) / n)


def plane_normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Unit normal of the plane through three points."""
    a, b, c = (np.asarray(x, dtype=np.float64) for x in (a, b, c))
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return np.zeros(3)
    return n / norm


# -------------------------------------------------------- torso-relative frame


def torso_frame(
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
    left_hip: np.ndarray,
    right_hip: np.ndarray,
) -> np.ndarray:
    """Build an orthonormal body-local frame as a 3x3 matrix of row vectors.

    Rows are (lateral, vertical, forward). Expressing landmarks in this frame
    makes every downstream measurement invariant to how the person is turned
    relative to the camera -- the property the old left-keypoint-only tracking
    completely lacked.
    """
    ls, rs, lh, rh = (
        np.asarray(p, dtype=np.float64)[:3] for p in (left_shoulder, right_shoulder, left_hip, right_hip)
    )
    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0

    lateral = rs - ls
    vertical = shoulder_mid - hip_mid

    ln = np.linalg.norm(lateral)
    vn = np.linalg.norm(vertical)
    if ln < 1e-9 or vn < 1e-9:
        return np.eye(3)
    lateral /= ln
    vertical /= vn

    forward = np.cross(lateral, vertical)
    fn = np.linalg.norm(forward)
    if fn < 1e-9:
        return np.eye(3)
    forward /= fn

    # Re-orthogonalize vertical against the other two (shoulders and hips are
    # rarely perfectly perpendicular on a real body).
    vertical = np.cross(forward, lateral)
    return np.stack([lateral, vertical, forward])


def to_torso_frame(points: np.ndarray, frame: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Express world points in the torso-local frame."""
    points = np.asarray(points, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)[:3]
    return (points[..., :3] - origin) @ frame.T
