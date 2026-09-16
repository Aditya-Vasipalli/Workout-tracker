"""Live spoken coaching: does it say the right thing, at the right rate?"""
import time

import pytest

from gymbro.coaching.policy import CONFIRMATIONS, CoachingPolicy
from gymbro.coaching.voice import NullSpeaker, Priority, VoiceCoach
from gymbro.engine.form import (
    FormReport, RuleResult, Severity, hip_sag, pelvic_control,
)
from gymbro.engine.repcounter import RepEvent


@pytest.fixture
def voice():
    speaker = NullSpeaker()
    v = VoiceCoach(speaker, default_cooldown_s=5.0, min_gap_s=0.0)
    v.start()
    yield v, speaker
    v.stop()


def said(speaker):
    return [t for _, t in speaker.spoken]


def result(name, passed, severity=Severity.FAULT, cue="fix it", direction=None):
    return RuleResult(name, passed, 1.0, "t", severity, cue, direction=direction)


class TestVoiceQueue:
    def test_does_not_block_the_caller(self, voice):
        """The capture loop must never wait on speech."""
        v, speaker = voice

        class SlowSpeaker(NullSpeaker):
            def speak(self, text):
                time.sleep(0.2)
                super().speak(text)

        v.speaker = SlowSpeaker()
        start = time.monotonic()
        for i in range(10):
            v.say(f"cue {i}", key=f"k{i}")
        assert time.monotonic() - start < 0.05, "say() blocked"

    def test_repeated_fault_speaks_once_per_cooldown(self, voice):
        v, speaker = voice
        for _ in range(100):
            v.say("Tuck your pelvis", key="pelvis")
        v.wait_idle()
        assert len(said(speaker)) == 1

    def test_different_cues_all_get_through(self, voice):
        v, speaker = voice
        v.say("one", key="a")
        v.say("two", key="b")
        v.say("three", key="c")
        v.wait_idle()
        assert len(said(speaker)) == 3

    def test_safety_jumps_the_queue(self):
        speaker = NullSpeaker()
        v = VoiceCoach(speaker, min_gap_s=0.0)
        # Queue without the worker running, so ordering is deterministic.
        v.say("count", Priority.REP_COUNT, key="c")
        v.say("tempo", Priority.TEMPO, key="t")
        v.say("danger", Priority.SAFETY, key="s")
        order = []
        while (cue := v._next_cue()) is not None:
            order.append(cue.text)
        assert order[0] == "danger"

    def test_stale_cues_are_dropped(self):
        """Advice about a position you left is worse than silence."""
        speaker = NullSpeaker()
        v = VoiceCoach(speaker, min_gap_s=0.0)
        v.say("old news", key="x", ttl_s=0.05)
        time.sleep(0.1)
        assert v._next_cue() is None
        assert v.dropped_stale == 1

    def test_interrupt_clears_the_queue(self):
        speaker = NullSpeaker()
        v = VoiceCoach(speaker, min_gap_s=0.0)
        v.say("a", Priority.FORM, key="a")
        v.say("b", Priority.FORM, key="b")
        v.interrupt("STOP", key="stop")
        remaining = []
        while (cue := v._next_cue()) is not None:
            remaining.append(cue.text)
        assert remaining == ["STOP"]

    def test_broken_engine_does_not_crash_the_workout(self, voice):
        v, _ = voice

        class Broken(NullSpeaker):
            def speak(self, text):
                raise RuntimeError("no audio device")

        v.speaker = Broken()
        v.say("hello", key="h")
        v.wait_idle()  # must not raise

    def test_disabled_coach_is_silent(self):
        speaker = NullSpeaker()
        v = VoiceCoach(speaker, enabled=False)
        assert v.say("anything") is False


class TestPolicy:
    def _report(self, *results):
        return FormReport(list(results))

    def test_momentary_fault_is_not_called(self, voice):
        """A single bad frame is jitter, not a fault worth interrupting for."""
        v, speaker = voice
        policy = CoachingPolicy(v)
        for _ in range(3):
            policy.on_frame(self._report(result("pelvic_control", False)))
        v.wait_idle()
        assert said(speaker) == []

    def test_sustained_fault_is_called(self, voice):
        v, speaker = voice
        policy = CoachingPolicy(v)
        for _ in range(policy.fault_threshold + 1):
            policy.on_frame(self._report(result("pelvic_control", False, cue="Tuck your pelvis")))
        v.wait_idle()
        assert "Tuck your pelvis" in said(speaker)

    def test_only_one_correction_at_a_time(self, voice):
        """Three simultaneous faults means acting on none. Say the worst."""
        v, speaker = voice
        policy = CoachingPolicy(v)
        for _ in range(policy.fault_threshold + 1):
            policy.on_frame(self._report(
                result("a", False, Severity.CUE, "minor thing"),
                result("b", False, Severity.UNSAFE, "dangerous thing"),
                result("c", False, Severity.FAULT, "medium thing"),
            ))
        v.wait_idle()
        assert said(speaker) == ["dangerous thing"]

    def test_persistent_ignoring_escalates(self, voice):
        v, speaker = voice
        policy = CoachingPolicy(v, correction_gap_s=0.0, cooldowns={
            Severity.UNSAFE: 0.0, Severity.FAULT: 0.0, Severity.CUE: 0.0,
        })
        for round_ in range(4):
            for _ in range(policy.fault_threshold + 1):
                policy.on_frame(self._report(
                    result("pelvic_control", False, cue="Tuck your pelvis under")
                ))
            v.forget("form:pelvic_control")
            time.sleep(0.02)
        v.wait_idle()
        spoken = said(speaker)
        assert any("Still arching" in s for s in spoken), spoken

    def test_fixing_a_fault_gets_acknowledged(self, voice):
        """Confirmation comes after a clean *rep*, not a run of clean frames."""
        v, speaker = voice
        policy = CoachingPolicy(v)
        rep = RepEvent(1, 0.0, 2.0, 0.95, 1.0, 1.0, 1.0, True, True)

        for _ in range(policy.fault_threshold + 1):
            policy.on_frame(self._report(result("hip_line", False, cue="Lift your hips")))
        policy.on_rep(rep, 1, 10)          # closes the faulty rep

        for _ in range(policy.recovery_frames):
            policy.on_frame(self._report(result("hip_line", True)))
        policy.on_rep(rep, 2, 10)          # closes a clean rep -> confirm

        v.wait_idle()
        assert any(c in said(speaker) for c in CONFIRMATIONS), said(speaker)

    def test_no_confirmation_mid_way_through_a_faulty_rep(self, voice):
        """The ascent of a bad rep passes a lockout check on the way up.

        Confirming there tells the user they fixed something while they are
        still doing it wrong.
        """
        v, speaker = voice
        policy = CoachingPolicy(v)
        for _ in range(policy.fault_threshold + 1):
            policy.on_frame(self._report(result("hip_line", False, cue="Lift your hips")))
        # Plenty of clean frames, but the rep never closes.
        for _ in range(policy.recovery_frames * 3):
            policy.on_frame(self._report(result("hip_line", True)))
        v.wait_idle()
        assert not any(c in said(speaker) for c in CONFIRMATIONS), said(speaker)

    def test_clean_form_stays_quiet(self, voice):
        v, speaker = voice
        policy = CoachingPolicy(v)
        for _ in range(200):
            policy.on_frame(self._report(result("a", True), result("b", True)))
        v.wait_idle()
        assert said(speaker) == []

    def test_skipped_rules_never_produce_cues(self, voice):
        """An invisible joint must not become a confident instruction."""
        v, speaker = voice
        policy = CoachingPolicy(v)
        skipped = RuleResult("x", True, float("nan"), "t", Severity.UNSAFE,
                             "do something", skipped=True, skip_reason="occluded")
        for _ in range(100):
            policy.on_frame(self._report(skipped))
        v.wait_idle()
        assert said(speaker) == []

    def test_rep_counts_are_announced(self, voice):
        v, speaker = voice
        policy = CoachingPolicy(v)
        rep = RepEvent(1, 0.0, 2.0, 0.95, 1.0, 1.0, 1.0, True, True)
        policy.on_rep(rep, count=3, target=10)
        v.wait_idle()
        assert "3" in said(speaker)

    def test_partial_rep_prompts_more_depth(self, voice):
        v, speaker = voice
        policy = CoachingPolicy(v)
        rep = RepEvent(1, 0.0, 2.0, 0.6, 1.0, 1.0, 0.0, False, True)
        policy.on_rep(rep, count=1, target=10)
        v.wait_idle()
        assert any("deeper" in s for s in said(speaker))

    def test_missed_hold_prompts_the_squeeze(self, voice):
        v, speaker = voice
        policy = CoachingPolicy(v)
        rep = RepEvent(1, 0.0, 2.0, 0.95, 1.0, 1.0, 0.1, True, False,
                       notes=["hold the squeeze: 0.1s of 1.0s"])
        policy.on_rep(rep, count=1, target=10)
        v.wait_idle()
        assert any("Hold the squeeze" in s for s in said(speaker))

    def test_shallow_rejection_is_explained(self, voice):
        v, speaker = voice
        policy = CoachingPolicy(v)
        policy.on_rejection("too shallow - you reached 55% of your range")
        v.wait_idle()
        assert any("Not deep enough" in s for s in said(speaker))

    def test_new_set_resets_escalation(self, voice):
        v, speaker = voice
        policy = CoachingPolicy(v)
        for _ in range(policy.fault_threshold + 1):
            policy.on_frame(self._report(result("pelvic_control", False)))
        policy.on_set_start("Hip Thrust", 10, None)
        assert policy._states == {}


    def test_cooldowns_are_tunable(self, voice):
        """How chatty the coach is should be a preference, not a constant."""
        v, speaker = voice
        quiet = CoachingPolicy(v, cooldowns={
            Severity.UNSAFE: 60.0, Severity.FAULT: 60.0, Severity.CUE: 60.0,
        })
        for _ in range(200):
            quiet.on_frame(self._report(result("a", False, cue="fix this")))
        v.wait_idle()
        assert len(said(speaker)) == 1


    def test_corrections_are_spaced_out(self, voice):
        """Two different faults must not be fired back to back."""
        v, speaker = voice
        policy = CoachingPolicy(v, correction_gap_s=30.0)
        for _ in range(60):
            policy.on_frame(self._report(
                result("a", False, Severity.FAULT, "first thing"),
                result("b", False, Severity.FAULT, "second thing"),
            ))
        v.wait_idle()
        assert len(said(speaker)) == 1, said(speaker)

    def test_safety_interrupts_the_gap(self, voice):
        """A dangerous position should not wait its turn."""
        v, speaker = voice
        policy = CoachingPolicy(v, correction_gap_s=30.0)
        for _ in range(15):
            policy.on_frame(self._report(result("a", False, Severity.FAULT, "minor")))
        for _ in range(15):
            policy.on_frame(self._report(
                result("a", False, Severity.FAULT, "minor"),
                result("danger", False, Severity.UNSAFE, "STOP - back rounding"),
            ))
        v.wait_idle()
        assert "STOP - back rounding" in said(speaker), said(speaker)

    def test_intermittent_peak_fault_accumulates_across_reps(self, voice):
        """A lockout fault gets only a few frames per rep and must still fire.

        Consecutive-frame counting made phase-scoped rules uncueable: the peak
        of a rep is shorter than the threshold, so the counter reset every rep.
        """
        v, speaker = voice
        policy = CoachingPolicy(v)
        for rep in range(4):
            for _ in range(6):   # peak frames, rule fails
                policy.on_frame(self._report(
                    result("pelvic_control", False, cue="Tuck your pelvis under")))
            # Between peaks the rule is phase-gated off: no results at all.
            for _ in range(20):
                policy.on_frame(self._report())
        v.wait_idle()
        assert "Tuck your pelvis under" in said(speaker), said(speaker)


class TestDirectionalCues:
    """The cue must say which way to move, and never the wrong way."""

    def test_sag_and_pike_get_opposite_cues(self):
        from tests.synthetic import glute_bridge_pose
        rule = hip_sag()
        sagging = rule.evaluate(glute_bridge_pose(200.0), {})
        piking = rule.evaluate(glute_bridge_pose(160.0), {})
        assert "lift your hips" in sagging.cue.lower()
        assert "drop your hips" in piking.cue.lower()
        assert sagging.cue != piking.cue

    def test_arching_tells_you_to_tuck(self):
        from tests.synthetic import glute_bridge_pose
        r = pelvic_control().evaluate(glute_bridge_pose(200.0), {"normalized": 0.9})
        assert r.direction == "high"
        assert "tuck" in r.cue.lower()

    def test_short_extension_tells_you_to_push_higher(self):
        from tests.synthetic import glute_bridge_pose
        r = pelvic_control().evaluate(glute_bridge_pose(150.0), {"normalized": 0.9})
        assert r.direction == "low"
        assert "higher" in r.cue.lower()
