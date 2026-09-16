"""Gravity alignment: the fix for a camera that isn't level."""
import numpy as np
import pytest

from gymbro.engine.form import torso_upright
from gymbro.engine.session import SetTracker
from gymbro.exercises.loader import get_exercise
from gymbro.pose.orientation import (
    Orientation, OrientationCalibrator, describe_tilt, identity,
)
from tests.synthetic import bridge_sequence, glute_bridge_pose, supine_rest_pose


def pitch_matrix(deg):
    """Camera pitched down by `deg`: rotates the world about the lateral axis."""
    t = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


def tilt(pose, deg):
    R = pitch_matrix(deg)
    out = glute_bridge_pose(160.0, timestamp=pose.timestamp)
    out.world = pose.world @ R.T
    out.confidence = pose.confidence
    return out


def upright_pose(timestamp=0.0):
    """A standing body: hips below shoulders, feet below hips."""
    j = {}
    for side, sx in (("left", -0.18), ("right", 0.18)):
        j[f"{side}_shoulder"] = (sx, 1.40, 0.0)
        j[f"{side}_hip"] = (sx * 0.8, 0.95, 0.0)
        j[f"{side}_knee"] = (sx * 0.8, 0.50, 0.02)
        j[f"{side}_ankle"] = (sx * 0.8, 0.08, 0.0)
        j[f"{side}_heel"] = (sx * 0.8, 0.04, -0.04)
        j[f"{side}_foot_index"] = (sx * 0.8, 0.03, 0.12)
        j[f"{side}_elbow"] = (sx * 1.1, 1.12, 0.0)
        j[f"{side}_wrist"] = (sx * 1.2, 0.88, 0.0)
    j["nose"] = (0.0, 1.58, 0.05)
    from tests.synthetic import make_pose
    return make_pose(j, timestamp=timestamp)


class TestTiltDetection:
    @pytest.mark.parametrize("deg", [0, 15, 30, 45, 60, 80])
    def test_standing_calibration_recovers_camera_pitch(self, deg):
        cal = OrientationCalibrator("standing", min_samples=5)
        for i in range(10):
            cal.add(tilt(upright_pose(i / 30), deg))
        result = cal.result()
        assert result is not None
        assert result.tilt_degrees == pytest.approx(deg, abs=1.0)
        assert result.reliable

    def test_holding_still_gives_high_confidence(self):
        cal = OrientationCalibrator("standing", min_samples=5)
        for i in range(20):
            cal.add(tilt(upright_pose(i / 30), 45))
        assert cal.stability > 0.99

    def test_moving_during_calibration_lowers_confidence(self):
        cal = OrientationCalibrator("standing", min_samples=5)
        rng = np.random.default_rng(0)
        for i in range(20):
            p = tilt(upright_pose(i / 30), 45)
            p.world = p.world + rng.normal(0, 0.15, p.world.shape)
            cal.add(p)
        assert cal.stability < 0.99

    @pytest.mark.parametrize("deg", [0, 20, 40, 55, 70, 85])
    def test_floor_method_works_lying_down(self, deg):
        """Someone in a cramped space may never stand up in frame.

        Measured against a rig that actually models floor contact, since that
        is what the calibration instruction asks the user to hold.
        """
        cal = OrientationCalibrator("floor", min_samples=5)
        for i in range(15):
            p = supine_rest_pose(i / 30)
            p.world = p.world @ pitch_matrix(deg).T
            cal.add(p)
        result = cal.result()
        assert result is not None
        assert result.tilt_degrees == pytest.approx(deg, abs=1.0)
        assert result.reliable

    def test_floor_method_survives_the_hips_moving(self):
        """The hips lift during a bridge, so the floor plane must not use them.

        An earlier version fitted the plane to the heels and hips and pivoted
        with every rep, producing a 93-degree reading for a 55-degree camera.
        """
        cal = OrientationCalibrator("floor", min_samples=10)
        for pose, _ in bridge_sequence(2):
            cal.add(tilt(pose, 55))
        assert cal.stability > 0.95

    def test_not_ready_before_enough_samples(self):
        cal = OrientationCalibrator("standing", min_samples=20)
        for i in range(5):
            cal.add(upright_pose(i / 30))
        assert not cal.ready and cal.result() is None

    def test_invalid_method_rejected(self):
        with pytest.raises(ValueError):
            OrientationCalibrator("vibes")


class TestCorrection:
    @pytest.mark.parametrize("deg", [0, 20, 40, 60, 75])
    def test_torso_upright_becomes_tilt_invariant(self, deg):
        """The bug: this measurement tracked camera pitch 1:1."""
        cal = OrientationCalibrator("standing", min_samples=5)
        for i in range(10):
            cal.add(tilt(upright_pose(i / 30), deg))
        orientation = cal.result()

        rule = torso_upright()
        level_truth = rule.evaluate(upright_pose(), {}).measured
        corrected = rule.evaluate(orientation.apply(tilt(upright_pose(), deg)), {}).measured
        assert corrected == pytest.approx(level_truth, abs=1.5), (
            f"at {deg} deg tilt: {corrected:.1f} vs {level_truth:.1f}"
        )

    def test_uncorrected_measurement_really_does_drift(self):
        """Documents the failure this module exists to fix."""
        rule = torso_upright()
        level = rule.evaluate(upright_pose(), {}).measured
        steep = rule.evaluate(tilt(upright_pose(), 60), {}).measured
        assert abs(steep - level) > 50

    def test_rep_counting_survives_a_steep_camera(self):
        exercise = get_exercise("glute_bridge")
        cal = OrientationCalibrator("floor", min_samples=5)
        for pose, t in bridge_sequence(1):
            cal.add(tilt(pose, 55))
        orientation = cal.result()

        tracker = SetTracker(exercise, target_reps=6, seeded_rom=(138.0, 176.0),
                             orientation=orientation)
        for pose, t in bridge_sequence(6):
            tracker.update(tilt(pose, 55), t)
        assert tracker.finish().counted_reps == 6

    def test_unreliable_calibration_is_not_applied(self):
        """Better to leave the pose alone than rotate it by a bad estimate."""
        bad = Orientation(up=np.array([0.0, 1.0, 0.0]), confidence=0.1, method="floor")
        pose = glute_bridge_pose(160.0)
        assert bad.apply(pose) is pose

    def test_identity_is_a_no_op(self):
        pose = glute_bridge_pose(160.0)
        assert np.allclose(identity().apply(pose).world, pose.world)

    def test_applied_pose_is_flagged(self):
        cal = OrientationCalibrator("standing", min_samples=5)
        for i in range(10):
            cal.add(tilt(upright_pose(i / 30), 30))
        out = cal.result().apply(glute_bridge_pose(160.0))
        assert out.meta["gravity_aligned"] is True
        assert out.meta["camera_tilt"] == pytest.approx(30, abs=1.0)


class TestReporting:
    @pytest.mark.parametrize("deg,phrase", [
        (5, "roughly level"), (25, "tilted"), (50, "steeply angled"), (80, "nearly overhead"),
    ])
    def test_describes_tilt_in_plain_language(self, deg, phrase):
        t = np.radians(deg)
        o = Orientation(up=np.array([0.0, np.cos(t), -np.sin(t)]), confidence=0.9, method="x")
        assert phrase in describe_tilt(o)

    def test_roundtrips_through_dict(self):
        o = Orientation(up=np.array([0.1, 0.9, -0.2]), confidence=0.8,
                        method="floor", samples=30)
        back = Orientation.from_dict(o.to_dict())
        assert np.allclose(back.up, o.up) and back.method == "floor"
