"""Canonical skeleton definition, backend-independent.

Backends (MediaPipe, MoveNet, ...) map their own keypoint sets onto this one so
nothing downstream depends on which model produced the pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

JOINTS: tuple[str, ...] = (
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)

JOINT_INDEX: dict[str, int] = {name: i for i, name in enumerate(JOINTS)}
NUM_JOINTS = len(JOINTS)

# Bones, for drawing and for bone-length consistency checks.
BONES: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"), ("left_heel", "left_foot_index"),
    ("right_ankle", "right_heel"), ("right_heel", "right_foot_index"),
)

MIRROR: dict[str, str] = {}
for _n in JOINTS:
    if _n.startswith("left_"):
        MIRROR[_n] = "right_" + _n[5:]
    elif _n.startswith("right_"):
        MIRROR[_n] = "left_" + _n[6:]
    else:
        MIRROR[_n] = _n


@dataclass
class Pose:
    """A single frame of pose.

    Attributes:
        world: (NUM_JOINTS, 3) metric coordinates in metres, hip-centred.
            Y is up, X is the subject's left-to-right, Z is toward the camera.
        pixels: (NUM_JOINTS, 2) coordinates in the *source* image, for drawing.
        confidence: (NUM_JOINTS,) visibility/presence in [0, 1].
        timestamp: seconds, monotonic.
    """

    world: np.ndarray
    pixels: np.ndarray
    confidence: np.ndarray
    timestamp: float
    source: str = "unknown"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.world = np.asarray(self.world, dtype=np.float64).reshape(NUM_JOINTS, 3)
        self.pixels = np.asarray(self.pixels, dtype=np.float64).reshape(NUM_JOINTS, 2)
        self.confidence = np.asarray(self.confidence, dtype=np.float64).reshape(NUM_JOINTS)

    def joint(self, name: str) -> np.ndarray:
        return self.world[JOINT_INDEX[name]]

    def conf(self, name: str) -> float:
        return float(self.confidence[JOINT_INDEX[name]])

    def visible(self, names: tuple[str, ...] | list[str], threshold: float = 0.5) -> bool:
        return all(self.conf(n) >= threshold for n in names)

    def mirrored(self) -> "Pose":
        """Left/right swapped, for training-side normalization."""
        idx = [JOINT_INDEX[MIRROR[n]] for n in JOINTS]
        world = self.world[idx].copy()
        world[:, 0] *= -1.0
        return Pose(
            world=world,
            pixels=self.pixels[idx].copy(),
            confidence=self.confidence[idx].copy(),
            timestamp=self.timestamp,
            source=self.source,
            meta=dict(self.meta),
        )


def bone_lengths(pose: Pose) -> dict[str, float]:
    """Length of each bone in metres. Used to detect implausible poses."""
    out = {}
    for a, b in BONES:
        out[f"{a}->{b}"] = float(np.linalg.norm(pose.joint(a) - pose.joint(b)))
    return out
