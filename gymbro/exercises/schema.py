"""Exercise specification: data, not code.

The old tracker hard-coded each exercise's thresholds inline and wrote bespoke
scoring functions per movement (`frog_pump_pelvic_score`, `deadlift_form_score`,
...). Adding an exercise meant editing a 3,400-line module. Here an exercise is
a YAML record, and the engine interprets it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..engine.form import (
    FormRule, Severity, hip_above_line, hip_sag, joint_angle_range, knee_tracking,
    limb_straight, pelvic_control, spine_neutral, stillness, symmetry, torso_upright,
)
from ..pose.geometry import angle_3d
from ..pose.skeleton import Pose


class Equipment(str, Enum):
    BODYWEIGHT = "bodyweight"
    DUMBBELL = "dumbbell"
    BAND = "band"
    BENCH = "bench"       # or sofa/bed edge
    MAT = "mat"


class Position(str, Enum):
    STANDING = "standing"
    SUPINE = "supine"          # on your back
    PRONE = "prone"            # face down
    SIDE_LYING = "side_lying"
    QUADRUPED = "quadruped"    # hands and knees
    SEATED = "seated"
    HINGE = "hinge"


class CameraView(str, Enum):
    SIDE = "side"
    FRONT = "front"
    ANGLED_45 = "angled_45"
    OVERHEAD = "overhead"


# ------------------------------------------------------------------ rep signal


@dataclass
class SignalSpec:
    """How to reduce a pose to the single scalar that drives rep counting.

    kind:
      joint_angle    - interior angle at `joints[1]`; higher = more extended
      joint_distance - 3D distance between two joints, in torso-normalized units
      height         - vertical (Y) position of a joint relative to a reference
      abduction      - angle of a limb away from the body's midline
    """

    kind: str
    joints: tuple[str, ...]
    reference: tuple[str, ...] = ()
    invert: bool = False        # flip so "higher = deeper into the rep"
    mirror_side: bool = True    # for unilateral work, follow the trained side

    def compute(self, pose: Pose, side: str | None = None) -> float:
        joints = self._resolve(self.joints, side)
        ref = self._resolve(self.reference, side)

        if self.kind == "joint_angle":
            value = angle_3d(*[pose.joint(j) for j in joints])

        elif self.kind == "joint_distance":
            raw = float(np.linalg.norm(pose.joint(joints[0]) - pose.joint(joints[1])))
            value = raw / max(self._torso_length(pose), 1e-6)

        elif self.kind == "height":
            y = float(pose.joint(joints[0])[1])
            if ref:
                y -= float(np.mean([pose.joint(j)[1] for j in ref]))
            value = y / max(self._torso_length(pose), 1e-6)

        elif self.kind == "abduction":
            # Angle between the limb and the body's long axis.
            hip = pose.joint(joints[0])
            distal = pose.joint(joints[1])
            shoulder = (pose.joint("left_shoulder") + pose.joint("right_shoulder")) / 2
            hip_mid = (pose.joint("left_hip") + pose.joint("right_hip")) / 2
            value = angle_3d(distal, hip, hip_mid + (hip_mid - shoulder))

        else:
            raise ValueError(f"unknown signal kind: {self.kind!r}")

        if not np.isfinite(value):
            return float("nan")
        return -value if self.invert else value

    def _resolve(self, joints: tuple[str, ...], side: str | None) -> tuple[str, ...]:
        """Substitute {side} placeholders for unilateral exercises."""
        if side is None or not self.mirror_side:
            return tuple(j.replace("{side}", "left") for j in joints)
        return tuple(j.replace("{side}", side) for j in joints)

    @staticmethod
    def _torso_length(pose: Pose) -> float:
        shoulder = (pose.joint("left_shoulder") + pose.joint("right_shoulder")) / 2
        hip = (pose.joint("left_hip") + pose.joint("right_hip")) / 2
        return float(np.linalg.norm(shoulder - hip))

    def required_joints(self, side: str | None = None) -> tuple[str, ...]:
        return tuple(set(self._resolve(self.joints, side) + self._resolve(self.reference, side)))


# ---------------------------------------------------------------- rep criteria


@dataclass
class RepCriteria:
    enter_threshold: float = 0.70
    exit_threshold: float = 0.30
    full_range_threshold: float = 0.85
    min_rep_s: float = 0.6
    max_rep_s: float = 20.0
    min_hold_s: float = 0.0
    # For isometrics (planks, wall sits) reps are seconds, not repetitions.
    is_isometric: bool = False


# ------------------------------------------------------------------- exercise


@dataclass
class Exercise:
    id: str
    name: str
    category: str
    primary_muscles: tuple[str, ...]
    equipment: tuple[Equipment, ...]
    position: Position
    signal: SignalSpec
    criteria: RepCriteria = field(default_factory=RepCriteria)
    rules: list[FormRule] = field(default_factory=list)

    unilateral: bool = False
    preferred_view: CameraView = CameraView.SIDE
    secondary_muscles: tuple[str, ...] = ()
    difficulty: str = "beginner"
    setup: str = ""
    cues: tuple[str, ...] = ()
    glute_emphasis: float = 0.0   # 0-1; drives glute-priority programming
    aliases: tuple[str, ...] = ()
    # Heavier variants this progresses into, easiest first.
    progressions: tuple[str, ...] = ()
    regressions: tuple[str, ...] = ()

    def required_joints(self, side: str | None = None) -> tuple[str, ...]:
        joints = set(self.signal.required_joints(side))
        for rule in self.rules:
            joints.update(rule.joints)
        return tuple(sorted(joints))

    @property
    def is_isometric(self) -> bool:
        return self.criteria.is_isometric


# ------------------------------------------------- declarative rule construction

_RULE_BUILDERS = {
    "joint_angle_range": joint_angle_range,
    "symmetry": symmetry,
    "knee_tracking": knee_tracking,
    "spine_neutral": spine_neutral,
    "hip_extension": hip_above_line,
    "torso_upright": torso_upright,
    "stillness": stillness,
    "pelvic_control": pelvic_control,
    "hip_sag": hip_sag,
    "limb_straight": limb_straight,
}


def build_rule(spec: dict) -> FormRule:
    """Instantiate a FormRule from its YAML representation."""
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in _RULE_BUILDERS:
        raise ValueError(f"unknown form rule type: {kind!r}")

    if "severity" in spec:
        spec["severity"] = Severity(spec["severity"])
    if spec.get("phase") not in (None, "any", "peak", "bottom", "moving"):
        raise ValueError(f"bad rule phase: {spec['phase']!r}")
    for key in ("triplet", "left_triplet", "right_triplet"):
        if key in spec:
            spec[key] = tuple(spec[key])

    return _RULE_BUILDERS[kind](**spec)
