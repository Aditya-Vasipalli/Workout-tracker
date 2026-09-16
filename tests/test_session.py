import numpy as np
import pytest

from gymbro.engine.session import SetTracker
from gymbro.exercises.loader import get_exercise
from tests.synthetic import bridge_sequence, glute_bridge_pose


def run_set(exercise_id, target_reps, sequence, **kw):
    tracker = SetTracker(get_exercise(exercise_id), target_reps=target_reps, **kw)
    feedback = []
    for pose, t in sequence:
        feedback.append(tracker.update(pose, t))
    return tracker.finish(), feedback


class TestEndToEnd:
    def test_counts_clean_bridges(self):
        result, _ = run_set(
            "glute_bridge", 8, bridge_sequence(8), seeded_rom=(138.0, 176.0)
        )
        assert result.counted_reps == 8
        assert result.good_reps == 8
        assert result.completed

    def test_form_score_is_high_on_clean_reps(self):
        result, _ = run_set(
            "glute_bridge", 6, bridge_sequence(6), seeded_rom=(138.0, 176.0)
        )
        assert result.form_score is not None
        assert result.form_score > 0.9, f"got {result.form_score}"
        assert result.form_coverage == 1.0

    def test_survives_realistic_keypoint_noise(self):
        """15mm of jitter must not change the rep count."""
        result, _ = run_set(
            "glute_bridge", 8,
            bridge_sequence(8, noise_m=0.015, seed=3),
            seeded_rom=(138.0, 176.0),
        )
        assert result.counted_reps == 8, f"noise broke counting: {result.counted_reps}"

    def test_view_invariance_the_old_tracker_lacked(self):
        """Same movement, body rotated 40 degrees: same rep count and score.

        This is the property 2D pixel angles cannot provide.
        """
        straight, _ = run_set(
            "glute_bridge", 6, bridge_sequence(6), seeded_rom=(138.0, 176.0)
        )
        rotated, _ = run_set(
            "glute_bridge", 6, bridge_sequence(6, yaw_deg=40.0),
            seeded_rom=(138.0, 176.0),
        )
        assert straight.counted_reps == rotated.counted_reps
        assert rotated.form_score == pytest.approx(straight.form_score, abs=0.02)

    def test_partial_range_reps_counted_but_not_good(self):
        """Deep enough to start a rep (79% of range) but short of full lockout."""
        result, _ = run_set(
            "glute_bridge", 6,
            bridge_sequence(6, top_deg=168.0),
            seeded_rom=(138.0, 176.0),
        )
        assert result.counted_reps == 6
        assert result.good_reps == 0
        assert not any(r.full_range for r in result.reps)

    def test_very_shallow_reps_explain_themselves(self):
        """Half-reps must not silently produce a count of zero with no reason."""
        result, feedback = run_set(
            "glute_bridge", 6,
            bridge_sequence(6, top_deg=155.0),
            seeded_rom=(138.0, 176.0),
        )
        assert result.counted_reps == 0
        assert result.shallow_attempts >= 4
        assert any("shallow" in w for w in result.warnings), result.warnings
        assert any("too shallow" in (f.message or "") for f in feedback)

    def test_skipping_the_hold_fails_tempo(self):
        result, _ = run_set(
            "glute_bridge", 5,
            bridge_sequence(5, hold_s=0.0, rep_period=4.0),
            seeded_rom=(138.0, 176.0),
        )
        assert result.counted_reps == 5
        assert result.good_reps == 0   # glute_bridge requires a 1s hold
        assert any("hold the squeeze" in n for r in result.reps for n in r.notes)

    def test_asymmetry_is_detected_and_cued(self):
        result, _ = run_set(
            "glute_bridge", 6,
            bridge_sequence(6, pelvis_tilt_deg=22.0),
            seeded_rom=(138.0, 176.0),
        )
        assert result.form_score is not None and result.form_score < 0.95
        assert any("pelvis" in c.lower() or "hip" in c.lower() for c in result.cues), result.cues

    def test_occlusion_is_reported_not_guessed(self):
        """Low confidence must produce a warning, never a confident score."""
        seq = [
            (glute_bridge_pose(a, timestamp=t, confidence=0.15), t)
            for a, t in ((150.0 + 10 * np.sin(i / 5), i / 30.0) for i in range(200))
        ]
        result, feedback = run_set("glute_bridge", 8, seq)
        assert result.counted_reps == 0
        assert result.tracking_quality == 0.0
        assert any("camera" in w or "tracked cleanly" in w for w in result.warnings)
        assert any(not f.tracking_ok for f in feedback)

    def test_missing_person_handled(self):
        result, feedback = run_set("glute_bridge", 5, [(None, i / 30.0) for i in range(60)])
        assert result.counted_reps == 0
        assert all("step into frame" in (f.message or "").lower() for f in feedback)

    def test_rom_learned_when_not_seeded(self):
        result, _ = run_set("glute_bridge", 10, bridge_sequence(10))
        assert result.rom_low is not None and result.rom_high is not None
        assert result.rom_low == pytest.approx(138.0, abs=1.5)
        assert result.rom_high == pytest.approx(176.0, abs=1.5)

    def test_unsafe_cue_surfaces_immediately(self):
        """RDL with a rounded spine should warn during the rep, not after."""
        tracker = SetTracker(get_exercise("romanian_deadlift"), target_reps=5)
        # Shoulder-hip-knee far from straight = flagged as rounding.
        pose = glute_bridge_pose(100.0, timestamp=0.0)
        fb = tracker.update(pose, 0.0)
        assert fb.form is not None
        assert any(r.name == "spine_neutral" for r in fb.form.results)

    def test_2d_backend_skips_depth_rules_rather_than_faking_them(self):
        pose = glute_bridge_pose(170.0)
        pose.meta["metric_3d"] = False
        tracker = SetTracker(get_exercise("banded_glute_bridge"), target_reps=5)
        fb = tracker.update(pose, 0.0)
        skipped = [r for r in fb.form.results if r.skipped]
        assert any("2D only" in r.skip_reason for r in skipped)


class TestSetResultIntegrity:
    def test_incomplete_set_not_marked_complete(self):
        result, _ = run_set(
            "glute_bridge", 12, bridge_sequence(5), seeded_rom=(138.0, 176.0)
        )
        assert result.counted_reps == 5 and not result.completed

    def test_good_reps_never_exceed_counted(self):
        for tilt in (0.0, 15.0, 30.0):
            r, _ = run_set(
                "glute_bridge", 8,
                bridge_sequence(8, pelvis_tilt_deg=tilt),
                seeded_rom=(138.0, 176.0),
            )
            assert r.good_reps <= r.counted_reps
