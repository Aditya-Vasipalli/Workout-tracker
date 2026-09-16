"""Decides what to say, and when.

Separated from `voice.py` (which decides *how* to say it) so the policy can be
tested without any audio at all.

The behaviour this encodes, learned from what makes spoken coaching tolerable
rather than maddening:

  * A fault must persist before it is called. Pose jitter and the natural
    bottom of a rep both produce single-frame "faults"; reacting to those means
    talking constantly and being wrong half the time.
  * One correction at a time. Given three simultaneous faults, a person can act
    on one. Saying all three means they act on none.
  * Escalate, don't repeat. The same words a third time read as a stuck record;
    a sharper phrasing reads as a coach noticing you ignored them.
  * Confirm the fix. Going quiet when someone corrects something is a missed
    opportunity -- a short "that's it" is what tells them the change was right.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from ..engine.form import FormReport, RuleResult, Severity
from ..engine.repcounter import RepEvent
from .voice import Priority, VoiceCoach

# A cue fires when a rule fails at least FAULT_THRESHOLD times within its last
# WINDOW_FRAMES *evaluated* frames.
#
# Counting consecutive failures instead -- the obvious approach -- silently
# breaks phase-scoped rules. A lockout check only runs at the top of each rep,
# so it might see seven failing frames per rep and reset in between; it could
# never reach a consecutive threshold of twelve no matter how badly or how
# often the lift was performed. Windowing over evaluated frames lets an
# intermittent fault accumulate across reps while still ignoring single-frame
# jitter.
WINDOW_FRAMES = 24
FAULT_THRESHOLD = 10

# A rule that fails during at least this many of the last REP_HISTORY reps is
# cued even if no single frame window crossed FAULT_THRESHOLD.
#
# Frame counting alone is not enough for faults that occur at one point in the
# rep. Arching at lockout might produce seven failing frames per rep among
# fifteen passing ones at the same phase -- a real fault on every single rep,
# but diluted below any frame threshold. Reps are the unit the fault actually
# recurs on, so they are the unit to count.
REP_HISTORY = 3
REP_FAULT_THRESHOLD = 2

# Frames of cleanliness before a previously-faulting rule is considered fixed.
RECOVERY_FRAMES = 20

# Minimum gap between any two corrections, regardless of which rules they came
# from. Without this, clearing one rule's window lets the next-worst fault fire
# on the very next frame, and the user gets a burst of three instructions they
# cannot act on. A safety-severity fault is allowed to cut this short.
CORRECTION_GAP_S = 3.0

_SEVERITY_PRIORITY = {
    Severity.UNSAFE: Priority.SAFETY,
    Severity.FAULT: Priority.FORM,
    Severity.CUE: Priority.TEMPO,
}

_COOLDOWN = {
    Severity.UNSAFE: 4.0,
    Severity.FAULT: 7.0,
    Severity.CUE: 10.0,
}

# Said after repeated failures to act on the original cue.
ESCALATIONS = {
    "pelvic_control": "Still arching. Reduce the range and squeeze instead of pushing higher.",
    "hip_line": "Hips again. Brace like someone's about to poke you in the stomach.",
    "spine_neutral": "Your back is still rounding. Stop the set and reset your position.",
    "knee_tracking_left": "Left knee is still caving. Slow down and push it out deliberately.",
    "knee_tracking_right": "Right knee is still caving. Slow down and push it out deliberately.",
    "hip_symmetry": "Still uneven. Press through both heels with the same force.",
    "torso_upright": "Still swinging. Drop the weight, this is too heavy right now.",
}

CONFIRMATIONS = ("That's it.", "Better.", "Good, hold that.", "Much better.")


@dataclass
class _RuleState:
    window: deque = field(default_factory=lambda: deque(maxlen=WINDOW_FRAMES))
    rep_history: deque = field(default_factory=lambda: deque(maxlen=REP_HISTORY))
    failed_this_rep: bool = False
    clean_frames: int = 0
    times_cued: int = 0
    was_cued: bool = False

    @property
    def failures(self) -> int:
        return sum(self.window)

    @property
    def faulty_reps(self) -> int:
        return sum(self.rep_history)

    def record(self, failed: bool) -> None:
        self.window.append(1 if failed else 0)
        if failed:
            self.failed_this_rep = True
        self.clean_frames = 0 if failed else self.clean_frames + 1

    def close_rep(self) -> None:
        self.rep_history.append(1 if self.failed_this_rep else 0)
        self.failed_this_rep = False

    def clear_window(self) -> None:
        self.window.clear()
        self.rep_history.clear()
        self.failed_this_rep = False


@dataclass
class CoachingPolicy:
    """Turns per-frame form reports into spoken cues.

    `cooldowns` sets how long before the same fault may be mentioned again,
    per severity. Raise them for a quieter coach, lower them for a pushier one.
    """

    voice: VoiceCoach
    fault_threshold: int = FAULT_THRESHOLD
    rep_fault_threshold: int = REP_FAULT_THRESHOLD
    recovery_frames: int = RECOVERY_FRAMES
    confirm_fixes: bool = True
    speak_rep_counts: bool = True
    cooldowns: dict = field(default_factory=lambda: dict(_COOLDOWN))
    correction_gap_s: float = CORRECTION_GAP_S

    _states: dict[str, _RuleState] = field(default_factory=dict)
    _confirm_index: int = 0
    _last_cued_rule: str | None = None
    _last_correction: float = 0.0

    # ----------------------------------------------------------- per frame

    def on_frame(self, report: FormReport | None, tracking_ok: bool = True) -> str | None:
        """Call once per frame. Returns the cue spoken, if any."""
        if report is None:
            return None

        evaluated = report.evaluated
        for result in evaluated:
            state = self._states.setdefault(result.name, _RuleState())
            state.record(not result.passed)

        return self._maybe_correct(evaluated)

    def _maybe_correct(self, results: list[RuleResult]) -> str | None:
        """Speak the single most important sustained fault."""
        candidates = [
            r for r in results
            if not r.passed and (
                self._states[r.name].failures >= self.fault_threshold
                or self._states[r.name].faulty_reps >= self.rep_fault_threshold
            )
        ]
        if not candidates:
            return None

        weight = {Severity.UNSAFE: 0, Severity.FAULT: 1, Severity.CUE: 2}
        candidates.sort(key=lambda r: (weight[r.severity], -self._states[r.name].failures))
        target = candidates[0]
        state = self._states[target.name]

        # One instruction at a time. Safety corrections may interrupt.
        since = time.monotonic() - self._last_correction
        if since < self.correction_gap_s and target.severity is not Severity.UNSAFE:
            return None

        text = target.cue
        if state.times_cued >= 2 and target.name in ESCALATIONS:
            text = ESCALATIONS[target.name]

        said = self.voice.say(
            text,
            priority=_SEVERITY_PRIORITY[target.severity],
            key=f"form:{target.name}",
            cooldown_s=self.cooldowns.get(target.severity, 7.0),
            ttl_s=2.0,
        )
        if said:
            state.times_cued += 1
            state.was_cued = True
            state.clear_window()
            self._last_cued_rule = target.name
            self._last_correction = time.monotonic()
            return text
        return None

    def _maybe_confirm(self) -> str | None:
        """Acknowledge a fault the user has corrected, judged per rep.

        Confirming on a run of clean *frames* fires far too early: the ascent
        of a still-faulty rep passes a lockout check on its way up, so the user
        gets told "that's it" in the middle of the very rep they are botching.
        A rep is the smallest unit over which "you fixed it" is meaningful.
        """
        for name, state in self._states.items():
            if (
                state.was_cued
                and len(state.rep_history) >= 1
                and state.rep_history[-1] == 0
            ):
                state.was_cued = False
                state.times_cued = 0
                text = CONFIRMATIONS[self._confirm_index % len(CONFIRMATIONS)]
                self._confirm_index += 1
                if self.voice.say(
                    text, priority=Priority.ENCOURAGE,
                    key=f"confirm:{name}", cooldown_s=8.0, ttl_s=1.2,
                ):
                    return text
        return None

    # -------------------------------------------------------------- events

    def on_rep(self, rep: RepEvent, count: int, target: int) -> None:
        """Announce a completed rep, and flag it if it was not a good one."""
        for state in self._states.values():
            state.close_rep()
        if self.confirm_fixes:
            self._maybe_confirm()

        if self.speak_rep_counts:
            self.voice.say(
                str(count), priority=Priority.REP_COUNT, key=f"rep:{count}",
                cooldown_s=0.0, ttl_s=1.2,
            )

        if not rep.full_range:
            self.voice.say(
                "Go deeper on the next one.", priority=Priority.TEMPO,
                key="rep:shallow", cooldown_s=12.0, ttl_s=2.5,
            )
        elif not rep.tempo_ok and rep.notes:
            if any("hold" in n for n in rep.notes):
                self.voice.say(
                    "Hold the squeeze at the top.", priority=Priority.TEMPO,
                    key="rep:hold", cooldown_s=12.0, ttl_s=2.5,
                )
            elif any("lowering" in n for n in rep.notes):
                self.voice.say(
                    "Slow the lowering down.", priority=Priority.TEMPO,
                    key="rep:eccentric", cooldown_s=12.0, ttl_s=2.5,
                )

    def on_rejection(self, reason: str) -> None:
        """A movement that did not count. Say why, sparingly."""
        if "shallow" in reason:
            text = "Not deep enough to count. Go further."
            key = "reject:shallow"
        elif "too fast" in reason:
            text = "Too fast to count. Slow it down and control it."
            key = "reject:fast"
        else:
            return
        self.voice.say(text, priority=Priority.TEMPO, key=key, cooldown_s=10.0, ttl_s=2.5)

    def on_tracking_lost(self, message: str) -> None:
        self.voice.say(
            message, priority=Priority.SAFETY, key="tracking",
            cooldown_s=8.0, ttl_s=3.0,
        )

    def on_set_start(self, exercise_name: str, target_reps: int, side: str | None) -> None:
        self.reset()
        where = f", {side} side" if side else ""
        self.voice.say(
            f"{exercise_name}{where}. {target_reps} reps.",
            priority=Priority.REP_COUNT, key="set:start", cooldown_s=0.0, ttl_s=6.0,
        )

    def on_set_end(self, counted: int, good: int, target: int) -> None:
        if counted >= target and good >= target:
            text = "Set complete. All clean."
        elif counted >= target:
            text = f"Set complete. {good} of {counted} were clean."
        else:
            text = f"Set stopped at {counted} of {target}."
        self.voice.say(
            text, priority=Priority.REP_COUNT, key="set:end", cooldown_s=0.0, ttl_s=6.0,
        )

    def reset(self) -> None:
        self._states.clear()
        self._last_cued_rule = None
        self._last_correction = 0.0
