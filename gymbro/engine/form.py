"""Declarative form rules evaluated on 3D pose.

Every rule is a named, testable predicate with an explicit tolerance and a
human-readable cue. This replaces the old scoring code, where numbers like
`* 0.3`, `max(5, ...)` and `form_score = 10` were invented and never validated
against anything.

Design rules:
  * A rule reports a *measurement* and a pass/fail against a stated tolerance,
    so a failure can always be explained in the units the user feels.
  * Rules that need depth declare `requires_3d` and are skipped (not guessed)
    when running on a 2D backend.
  * Rules that need a joint the model cannot see are skipped, and the skip is
    surfaced -- never silently treated as a pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from ..pose.geometry import angle_3d, point_line_distance, to_torso_frame, torso_frame
from ..pose.skeleton import Pose


class Severity(str, Enum):
    CUE = "cue"            # a refinement; rep still counts
    FAULT = "fault"        # meaningful breakdown; rep counts but flagged
    UNSAFE = "unsafe"      # stop-the-set territory


@dataclass
class RuleResult:
    name: str
    passed: bool
    measured: float
    target: str
    severity: Severity
    cue: str
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class FormRule:
    """One checkable aspect of technique."""

    name: str
    joints: tuple[str, ...]
    check: Callable[[Pose, dict], tuple[bool, float]]
    target: str
    cue: str
    severity: Severity = Severity.FAULT
    requires_3d: bool = False
    # Only evaluate during this phase of the rep, if set.
    phase: str | None = None

    def evaluate(self, pose: Pose, context: dict | None = None) -> RuleResult:
        context = context or {}

        if self.requires_3d and pose.meta.get("metric_3d") is False:
            return RuleResult(
                self.name, True, float("nan"), self.target, self.severity, self.cue,
                skipped=True,
                skip_reason="needs metric 3D; current backend is 2D only",
            )

        if not pose.visible(self.joints, threshold=0.5):
            missing = [j for j in self.joints if pose.conf(j) < 0.5]
            return RuleResult(
                self.name, True, float("nan"), self.target, self.severity, self.cue,
                skipped=True,
                skip_reason=f"can't see {', '.join(missing)} - adjust camera",
            )

        try:
            passed, measured = self.check(pose, context)
        except Exception as exc:  # a broken rule must not kill the session
            return RuleResult(
                self.name, True, float("nan"), self.target, self.severity, self.cue,
                skipped=True, skip_reason=f"rule error: {exc}",
            )

        return RuleResult(
            self.name, bool(passed), float(measured), self.target, self.severity, self.cue
        )


# --------------------------------------------------------------- rule builders
# These are parameterized factories so the exercise library can declare rules as
# data rather than each exercise carrying bespoke scoring code.


def joint_angle_range(
    name: str,
    triplet: tuple[str, str, str],
    lo: float,
    hi: float,
    cue: str,
    severity: Severity = Severity.FAULT,
    phase: str | None = None,
) -> FormRule:
    """Require a joint angle to sit within [lo, hi] degrees."""

    def check(pose: Pose, ctx: dict) -> tuple[bool, float]:
        a = angle_3d(pose.joint(triplet[0]), pose.joint(triplet[1]), pose.joint(triplet[2]))
        return (lo <= a <= hi, a)

    return FormRule(
        name=name, joints=triplet, check=check,
        target=f"{lo:.0f}-{hi:.0f} deg", cue=cue, severity=severity, phase=phase,
    )


def symmetry(
    name: str,
    left_triplet: tuple[str, str, str],
    right_triplet: tuple[str, str, str],
    max_diff_deg: float,
    cue: str,
    severity: Severity = Severity.FAULT,
) -> FormRule:
    """Require left and right joint angles to stay within `max_diff_deg`.

    Catches the single most common glute-training error: one side taking over.
    """

    def check(pose: Pose, ctx: dict) -> tuple[bool, float]:
        left = angle_3d(*[pose.joint(j) for j in left_triplet])
        right = angle_3d(*[pose.joint(j) for j in right_triplet])
        diff = abs(left - right)
        return (diff <= max_diff_deg, diff)

    return FormRule(
        name=name, joints=tuple(set(left_triplet + right_triplet)), check=check,
        target=f"<={max_diff_deg:.0f} deg apart", cue=cue, severity=severity,
    )


def knee_tracking(
    name: str = "knee_tracking",
    side: str = "left",
    max_deviation_m: float = 0.08,
    cue: str = "Drive your knee out over your middle toe - don't let it cave in",
    severity: Severity = Severity.UNSAFE,
) -> FormRule:
    """Knee should stay over the foot, not collapse medially (valgus).

    Measured as the 3D distance from the knee to the hip-ankle line, which is
    genuinely impossible to assess from a single 2D front view when the subject
    is rotated -- hence requires_3d.
    """
    joints = (f"{side}_hip", f"{side}_knee", f"{side}_ankle")

    def check(pose: Pose, ctx: dict) -> tuple[bool, float]:
        dev = point_line_distance(
            pose.joint(joints[1]), pose.joint(joints[0]), pose.joint(joints[2])
        )
        return (dev <= max_deviation_m, dev)

    return FormRule(
        name=f"{name}_{side}", joints=joints, check=check,
        target=f"<={max_deviation_m * 100:.0f}cm from hip-ankle line",
        cue=cue, severity=severity, requires_3d=True,
    )


def spine_neutral(
    name: str = "spine_neutral",
    max_flexion_deg: float = 20.0,
    cue: str = "Ribs down, long spine - stop rounding your lower back",
    severity: Severity = Severity.UNSAFE,
) -> FormRule:
    """Shoulder-hip-knee should stay close to a straight line through the trunk.

    This is a coarse proxy for lumbar flexion: MediaPipe has no spine landmarks,
    so we cannot measure true lumbar curvature. It catches gross rounding, not
    subtle positioning -- stated plainly rather than overclaimed.
    """
    joints = ("left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee")

    def check(pose: Pose, ctx: dict) -> tuple[bool, float]:
        shoulder = (pose.joint("left_shoulder") + pose.joint("right_shoulder")) / 2
        hip = (pose.joint("left_hip") + pose.joint("right_hip")) / 2
        knee = (pose.joint("left_knee") + pose.joint("right_knee")) / 2
        deviation = 180.0 - angle_3d(shoulder, hip, knee)
        return (deviation <= max_flexion_deg, deviation)

    return FormRule(
        name=name, joints=joints, check=check,
        target=f"<={max_flexion_deg:.0f} deg from neutral", cue=cue,
        severity=severity, requires_3d=True,
    )


def hip_above_line(
    name: str = "hip_extension",
    min_angle_deg: float = 165.0,
    cue: str = "Squeeze your glutes and finish the lockout - shoulder, hip, knee in one line",
    severity: Severity = Severity.CUE,
) -> FormRule:
    """Full hip extension at the top of a bridge/thrust."""
    joints = ("left_shoulder", "left_hip", "left_knee")

    def check(pose: Pose, ctx: dict) -> tuple[bool, float]:
        shoulder = (pose.joint("left_shoulder") + pose.joint("right_shoulder")) / 2
        hip = (pose.joint("left_hip") + pose.joint("right_hip")) / 2
        knee = (pose.joint("left_knee") + pose.joint("right_knee")) / 2
        a = angle_3d(shoulder, hip, knee)
        return (a >= min_angle_deg, a)

    return FormRule(
        name=name, joints=("left_shoulder", "right_shoulder", "left_hip", "right_hip",
                           "left_knee", "right_knee"),
        check=check, target=f">={min_angle_deg:.0f} deg", cue=cue, severity=severity,
    )


def torso_upright(
    name: str = "torso_upright",
    max_lean_deg: float = 25.0,
    cue: str = "Chest up - you're tipping forward",
    severity: Severity = Severity.FAULT,
) -> FormRule:
    """Limit forward trunk lean, measured against gravity (world Y axis)."""
    joints = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")

    def check(pose: Pose, ctx: dict) -> tuple[bool, float]:
        shoulder = (pose.joint("left_shoulder") + pose.joint("right_shoulder")) / 2
        hip = (pose.joint("left_hip") + pose.joint("right_hip")) / 2
        trunk = shoulder - hip
        n = np.linalg.norm(trunk)
        if n < 1e-6:
            return (True, 0.0)
        lean = float(np.degrees(np.arccos(np.clip(trunk[1] / n, -1.0, 1.0))))
        return (lean <= max_lean_deg, lean)

    return FormRule(
        name=name, joints=joints, check=check,
        target=f"<={max_lean_deg:.0f} deg lean", cue=cue, severity=severity,
    )


def stillness(
    name: str,
    joint: str,
    max_drift_m: float,
    cue: str,
    severity: Severity = Severity.CUE,
) -> FormRule:
    """A joint that should not move (e.g. the hips during a clamshell).

    Reads the reference position from context, which the session engine seeds at
    the start of the set.
    """

    def check(pose: Pose, ctx: dict) -> tuple[bool, float]:
        ref = ctx.get("reference_positions", {}).get(joint)
        if ref is None:
            return (True, 0.0)
        drift = float(np.linalg.norm(pose.joint(joint) - np.asarray(ref)))
        return (drift <= max_drift_m, drift)

    return FormRule(
        name=name, joints=(joint,), check=check,
        target=f"<={max_drift_m * 100:.0f}cm drift", cue=cue,
        severity=severity, requires_3d=True,
    )


# ------------------------------------------------------------------ evaluation


@dataclass
class FormReport:
    """Aggregated rule results over a rep or a set."""

    results: list[RuleResult] = field(default_factory=list)

    @property
    def evaluated(self) -> list[RuleResult]:
        return [r for r in self.results if not r.skipped]

    @property
    def skipped(self) -> list[RuleResult]:
        return [r for r in self.results if r.skipped]

    @property
    def failures(self) -> list[RuleResult]:
        return [r for r in self.evaluated if not r.passed]

    @property
    def unsafe(self) -> list[RuleResult]:
        return [r for r in self.failures if r.severity is Severity.UNSAFE]

    @property
    def coverage(self) -> float:
        """Fraction of rules actually checked. Low coverage = don't trust the score."""
        if not self.results:
            return 0.0
        return len(self.evaluated) / len(self.results)

    def score(self) -> float | None:
        """Fraction of evaluated rules passed, or None if nothing was checkable.

        Deliberately returns None rather than a number when coverage is too low.
        A confident-looking percentage derived from two visible joints is worse
        than admitting the camera couldn't see enough.
        """
        evaluated = self.evaluated
        if not evaluated or self.coverage < 0.5:
            return None

        weights = {Severity.CUE: 1.0, Severity.FAULT: 2.0, Severity.UNSAFE: 4.0}
        total = sum(weights[r.severity] for r in evaluated)
        earned = sum(weights[r.severity] for r in evaluated if r.passed)
        return earned / total if total else None

    def primary_cue(self) -> str | None:
        """The single most important thing to say right now."""
        for severity in (Severity.UNSAFE, Severity.FAULT, Severity.CUE):
            for r in self.failures:
                if r.severity is severity:
                    return r.cue
        return None


def evaluate_rules(rules: list[FormRule], pose: Pose, context: dict | None = None) -> FormReport:
    return FormReport([rule.evaluate(pose, context) for rule in rules])
