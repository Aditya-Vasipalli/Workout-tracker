"""Per-set tracking: pose in, verified reps and coaching cues out.

This is the piece the old tracker spread across `track_exercise` (660 lines),
`display_dual_camera_feed` and `run_workout`. Keeping it separate from any
camera or display code means the whole thing can be driven from recorded poses
in a test, which is how the rep logic gets verified without a human in front of
a webcam.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..exercises.schema import Exercise
from ..pose.filters import KeypointFilter
from ..pose.skeleton import NUM_JOINTS, Pose
from .form import FormReport, RuleResult, Severity, evaluate_rules
from .repcounter import RepCounter, RepEvent


@dataclass
class SetResult:
    exercise_id: str
    side: str | None
    target_reps: int
    counted_reps: int
    good_reps: int
    partial_reps: int
    form_score: float | None
    form_coverage: float
    duration_s: float
    completed: bool
    reps: list[RepEvent] = field(default_factory=list)
    cues: list[str] = field(default_factory=list)
    rom_low: float | None = None
    rom_high: float | None = None
    tracking_quality: float = 0.0
    warnings: list[str] = field(default_factory=list)
    shallow_attempts: int = 0


class SetTracker:
    """Drives one set of one exercise.

    Feed it poses with `update()`. It returns a `FrameFeedback` each frame
    describing what to show the user right now, and accumulates a `SetResult`.
    """

    def __init__(
        self,
        exercise: Exercise,
        target_reps: int,
        side: str | None = None,
        seeded_rom: tuple[float, float] | None = None,
        smoothing: bool = True,
    ) -> None:
        self.exercise = exercise
        self.target_reps = target_reps
        self.side = side if exercise.unilateral else None

        c = exercise.criteria
        self.counter = RepCounter(
            enter_threshold=c.enter_threshold,
            exit_threshold=c.exit_threshold,
            full_range_threshold=c.full_range_threshold,
            min_rep_s=c.min_rep_s,
            max_rep_s=c.max_rep_s,
            min_hold_s=c.min_hold_s,
        )
        if seeded_rom:
            self.counter.seed_calibration(*seeded_rom)

        self._filter = KeypointFilter(NUM_JOINTS) if smoothing else None
        self._rule_results: list[RuleResult] = []
        self._context: dict = {"reference_positions": {}}
        self._start_time: float | None = None
        self._last_time: float | None = None
        self._frames = 0
        self._tracked_frames = 0
        self._required = exercise.required_joints(self.side)

    # ------------------------------------------------------------ per frame

    def update(self, pose: Pose | None, timestamp: float) -> "FrameFeedback":
        self._frames += 1
        if self._start_time is None:
            self._start_time = timestamp
        self._last_time = timestamp

        if pose is None:
            return FrameFeedback(
                rep_completed=None, phase=self.counter.phase.value,
                rep_count=self.counter.count, signal=float("nan"),
                message="No one detected - step into frame", tracking_ok=False,
            )

        if self._filter is not None:
            smoothed = self._filter(pose.world, timestamp, pose.confidence)
            pose = Pose(
                world=smoothed, pixels=pose.pixels, confidence=pose.confidence,
                timestamp=pose.timestamp, source=pose.source, meta=pose.meta,
            )

        # Are the joints this exercise depends on actually visible?
        if not pose.visible(self._required, threshold=0.4):
            missing = [j for j in self._required if pose.conf(j) < 0.4]
            return FrameFeedback(
                rep_completed=None, phase=self.counter.phase.value,
                rep_count=self.counter.count, signal=float("nan"),
                message=f"Can't see your {_humanize(missing[0])} - adjust the camera",
                tracking_ok=False,
            )

        self._tracked_frames += 1
        self._seed_reference_positions(pose)

        signal = self.exercise.signal.compute(pose, self.side)

        # Publish where we are in the rep so phase-scoped rules (lockout checks
        # and the like) only fire where they are meaningful.
        cal = self.counter.calibration
        self._context["normalized"] = cal.normalize(signal) if cal else None
        self._context["phase"] = self.counter.phase.value

        report = evaluate_rules(self.exercise.rules, pose, self._context)
        self._rule_results.extend(report.results)

        rep = self.counter.update(signal, timestamp)

        # Unsafe form is worth interrupting for, mid-rep. Otherwise, explain
        # why a movement just failed to count before offering a technique cue.
        message = report.primary_cue()
        recent = self.counter.latest_rejection(since=timestamp - 2.0)
        if recent:
            message = recent
        if report.unsafe:
            message = report.unsafe[0].cue

        return FrameFeedback(
            rep_completed=rep,
            phase=self.counter.phase.value,
            rep_count=self.counter.count,
            signal=signal,
            normalized=(
                self.counter.calibration.normalize(signal)
                if self.counter.calibration else 0.0
            ),
            message=message,
            tracking_ok=True,
            form=report,
        )

    def _seed_reference_positions(self, pose: Pose) -> None:
        """Record where 'still' joints started, for stillness rules."""
        refs = self._context["reference_positions"]
        for rule in self.exercise.rules:
            if rule.name.endswith(("_stable", "_still")) and rule.joints:
                joint = rule.joints[0]
                if joint not in refs and pose.conf(joint) >= 0.5:
                    refs[joint] = pose.joint(joint).copy()

    # --------------------------------------------------------------- result

    @property
    def done(self) -> bool:
        return self.counter.count >= self.target_reps

    def finish(self) -> SetResult:
        report = FormReport(self._rule_results)
        duration = (
            (self._last_time - self._start_time)
            if self._start_time is not None and self._last_time is not None else 0.0
        )
        quality = self._tracked_frames / self._frames if self._frames else 0.0

        warnings: list[str] = []
        if quality < 0.6:
            warnings.append(
                f"Only {quality * 100:.0f}% of frames tracked cleanly - "
                f"reposition the camera before trusting these numbers."
            )
        score = report.score()
        if score is None and self._rule_results:
            warnings.append(
                "Not enough of your body was visible to score form for this set."
            )

        cal = self.counter.calibration
        cues = _summarize_cues(report)

        # If movement happened but little of it counted, say why.
        if self.counter.shallow_attempts >= max(2, self.target_reps // 3):
            warnings.append(
                f"{self.counter.shallow_attempts} movements were too shallow to "
                f"count. Either go deeper, or if that is your honest full range, "
                f"re-run calibration for this exercise."
            )
        for reason in self.counter.rejection_summary()[:2]:
            if reason not in cues:
                cues.append(reason)

        return SetResult(
            exercise_id=self.exercise.id,
            side=self.side,
            target_reps=self.target_reps,
            counted_reps=self.counter.count,
            good_reps=self.counter.good_count,
            partial_reps=self.counter.partial_count,
            form_score=score,
            form_coverage=report.coverage,
            duration_s=duration,
            completed=self.counter.count >= self.target_reps,
            reps=list(self.counter.reps),
            cues=cues,
            rom_low=cal.low if cal else None,
            rom_high=cal.high if cal else None,
            tracking_quality=quality,
            warnings=warnings,
            shallow_attempts=self.counter.shallow_attempts,
        )


@dataclass
class FrameFeedback:
    rep_completed: RepEvent | None
    phase: str
    rep_count: int
    signal: float
    message: str | None = None
    tracking_ok: bool = True
    normalized: float = 0.0
    form: FormReport | None = None


def _summarize_cues(report: FormReport, limit: int = 3) -> list[str]:
    """The most frequent failures across the set, worst severity first."""
    counts: dict[str, tuple[int, Severity, str]] = {}
    totals: dict[str, int] = {}

    for r in report.results:
        if r.skipped:
            continue
        totals[r.name] = totals.get(r.name, 0) + 1
        if not r.passed:
            n, _, _ = counts.get(r.name, (0, r.severity, r.cue))
            counts[r.name] = (n + 1, r.severity, r.cue)

    weight = {Severity.UNSAFE: 3, Severity.FAULT: 2, Severity.CUE: 1}
    ranked = sorted(
        counts.items(),
        key=lambda kv: (-weight[kv[1][1]], -(kv[1][0] / max(totals.get(kv[0], 1), 1))),
    )

    out = []
    for name, (fails, _sev, cue) in ranked[:limit]:
        pct = fails / max(totals.get(name, 1), 1) * 100
        if pct >= 20:  # ignore a couple of noisy frames
            out.append(f"{cue} ({pct:.0f}% of frames)")
    return out


def _humanize(joint: str) -> str:
    return joint.replace("_", " ")
