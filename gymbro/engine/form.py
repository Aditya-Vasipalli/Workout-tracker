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

from ..pose.geometry import (
    angle_3d, point_line_distance, signed_angle_about_axis, to_torso_frame, torso_frame,
)
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
    # "high" = measurement above the target band, "low" = below. Lets the
    # spoken cue say which way to move instead of only that something is off.
    direction: str | None = None


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
    # Direction-specific corrections. "Your knee angle is wrong" is useless;
    # "walk your feet closer" tells you what to actually do.
    cue_high: str = ""
    cue_low: str = ""

    def resolve_cue(self, direction: str | None) -> str:
        if direction == "high" and self.cue_high:
            return self.cue_high
        if direction == "low" and self.cue_low:
            return self.cue_low
        return self.cue

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
            outcome = self.check(pose, context)
        except Exception as exc:  # a broken rule must not kill the session
            return RuleResult(
                self.name, True, float("nan"), self.target, self.severity, self.cue,
                skipped=True, skip_reason=f"rule error: {exc}",
            )

        if len(outcome) == 3:
            passed, measured, direction = outcome
        else:
            passed, measured = outcome
            direction = None

        return RuleResult(
            self.name, bool(passed), float(measured), self.target, self.severity,
            self.resolve_cue(direction), direction=direction,
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
    cue_high: str = "",
    cue_low: str = "",
) -> FormRule:
    """Require a joint angle to sit within [lo, hi] degrees."""

    def check(pose: Pose, ctx: dict):
        a = angle_3d(pose.joint(triplet[0]), pose.joint(triplet[1]), pose.joint(triplet[2]))
        if a > hi:
            return (False, a, "high")
        if a < lo:
            return (False, a, "low")
        return (True, a, None)

    return FormRule(
        name=name, joints=triplet, check=check,
        target=f"{lo:.0f}-{hi:.0f} deg", cue=cue, severity=severity, phase=phase,
        cue_high=cue_high, cue_low=cue_low,
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
    phase: str | None = "peak",
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
        phase=phase,
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


def hip_line_angle(pose: Pose) -> float:
    """Shoulder-hip-knee angle on a continuous 0-360 scale.

    `angle_3d` returns an interior angle capped at 180, so it reports 165 for
    both a 165-degree hip (under-extended) and a 195-degree one (hyperextended
    past straight). Those are opposite faults needing opposite corrections --
    telling someone to lift their hips when they are already piking is worse
    than saying nothing.

    The sign comes from rotation about the anatomical lateral axis
    (left hip -> right hip), which is defined by landmarks rather than by world
    orientation, so it stays consistent whether the subject is supine, prone or
    standing. Below 180 is hip flexion; above 180 is hyperextension.
    """
    shoulder = (pose.joint("left_shoulder") + pose.joint("right_shoulder")) / 2
    hip = (pose.joint("left_hip") + pose.joint("right_hip")) / 2
    knee = (pose.joint("left_knee") + pose.joint("right_knee")) / 2
    axis = pose.joint("right_hip") - pose.joint("left_hip")

    signed = signed_angle_about_axis(shoulder, hip, knee, axis)
    if not np.isfinite(signed):
        return float("nan")
    return signed if signed >= 0 else 360.0 + signed


def pelvic_control(
    name: str = "pelvic_control",
    neutral_lo: float = 160.0,
    neutral_hi: float = 186.0,
    severity: Severity = Severity.FAULT,
    phase: str | None = "peak",
) -> FormRule:
    """Catch hip extension achieved by arching the back rather than the glutes.

    An honest note on what this measures. MediaPipe has no pelvis or spine
    landmarks -- no ASIS, no sacrum -- so true pelvic tilt is not observable.
    What *is* observable is the shoulder-hip-knee angle overshooting a straight
    line at the top of a bridge or thrust, which is what happens when someone
    runs out of hip extension and borrows the rest from lumbar extension. It is
    a proxy for the fault, not a measurement of pelvic angle, and it catches the
    gross version rather than a few degrees.
    """
    joints = ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
              "left_knee", "right_knee")

    def check(pose: Pose, ctx: dict):
        a = hip_line_angle(pose)
        if not np.isfinite(a):
            return (True, a, None)
        if a > neutral_hi:
            return (False, a, "high")
        if a < neutral_lo:
            return (False, a, "low")
        return (True, a, None)

    return FormRule(
        name=name, joints=joints, check=check,
        target=f"{neutral_lo:.0f}-{neutral_hi:.0f} deg",
        cue="Keep your hips in line with your shoulders and knees",
        cue_high="Tuck your pelvis under and drop your ribs - you're arching your back, not squeezing your glutes",
        cue_low="Push your hips higher and squeeze your glutes at the top",
        severity=severity, requires_3d=True, phase=phase,
    )


def hip_sag(
    name: str = "hip_line",
    max_sag_deg: float = 12.0,
    severity: Severity = Severity.FAULT,
) -> FormRule:
    """Hips dropping or piking in a plank-type hold.

    The spoken cue is "brace your core", because that is the correction, but
    what is measured is the hip line -- muscle activation is not observable
    from pose. Saying it this way is the honest version of a core cue.
    """
    joints = ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
              "left_knee", "right_knee")

    def check(pose: Pose, ctx: dict):
        a = hip_line_angle(pose)
        if not np.isfinite(a):
            return (True, a, None)
        # Positive deviation = hyperextension (hips dropped through the line);
        # negative = flexion (hips piked up).
        deviation = a - 180.0
        if deviation > max_sag_deg:
            return (False, deviation, "high")
        if deviation < -max_sag_deg:
            return (False, deviation, "low")
        return (True, deviation, None)

    return FormRule(
        name=name, joints=joints, check=check,
        target=f"within {max_sag_deg:.0f} deg of a straight line",
        cue="Straighten your body into one line",
        cue_high="Brace your core and lift your hips - they're sagging",
        cue_low="Drop your hips - you're piking up too high",
        severity=severity, requires_3d=True,
    )


def limb_straight(
    name: str,
    triplet: tuple[str, str, str],
    min_angle_deg: float = 160.0,
    cue: str = "Straighten that limb",
    severity: Severity = Severity.CUE,
    phase: str | None = None,
) -> FormRule:
    """A limb that should be extended (bird dog, swimming, leg circles)."""

    def check(pose: Pose, ctx: dict):
        a = angle_3d(*[pose.joint(j) for j in triplet])
        return (a >= min_angle_deg, a, None if a >= min_angle_deg else "low")

    return FormRule(
        name=name, joints=triplet, check=check,
        target=f">={min_angle_deg:.0f} deg", cue=cue,
        cue_low=cue, severity=severity, phase=phase,
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


def rule_applies(rule: FormRule, context: dict) -> bool:
    """Whether a phase-scoped rule should run on this frame.

    A lockout check must only fire at the top of the rep -- evaluating it at the
    bottom, where failing it is simply what the bottom of a rep looks like,
    would drag the score down for correct technique.
    """
    if rule.phase in (None, "any"):
        return True
    norm = context.get("normalized")
    if norm is None:
        return True
    if rule.phase == "peak":
        return norm >= 0.80
    if rule.phase == "bottom":
        return norm <= 0.20
    if rule.phase == "moving":
        return 0.20 < norm < 0.80
    return True


def evaluate_rules(rules: list[FormRule], pose: Pose, context: dict | None = None) -> FormReport:
    context = context or {}
    return FormReport([
        rule.evaluate(pose, context) for rule in rules if rule_applies(rule, context)
    ])
