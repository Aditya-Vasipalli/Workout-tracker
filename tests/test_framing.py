"""Framing guidance for a camera that can't be moved much."""
import numpy as np
import pytest

from gymbro.pose.framing import check_framing
from gymbro.pose.skeleton import JOINT_INDEX, NUM_JOINTS, Pose

REQUIRED = ("left_shoulder", "left_hip", "left_knee", "left_ankle")


def framed_pose(width=640, height=480, fill=0.6, confidence=0.9):
    """A pose laid out sensibly inside the frame.

    Every joint gets a position: leaving some at (0, 0) while still marking
    them confident would stretch the body's bounding box to the frame corner
    and make the fill measurement meaningless.
    """
    world = np.zeros((NUM_JOINTS, 3))
    pixels = np.zeros((NUM_JOINTS, 2))
    top, bottom = height * (0.5 - fill / 2), height * (0.5 + fill / 2)

    # Head-to-toe chain down the middle of the frame.
    chain = ("nose", "left_shoulder", "left_elbow", "left_wrist", "left_hip",
             "left_knee", "left_ankle", "left_heel", "left_foot_index")
    for i, name in enumerate(chain):
        pixels[JOINT_INDEX[name]] = (
            width * 0.5, top + (bottom - top) * i / (len(chain) - 1)
        )
    for name in ("right_shoulder", "right_elbow", "right_wrist", "right_hip",
                 "right_knee", "right_ankle", "right_heel", "right_foot_index"):
        mirror = name.replace("right", "left")
        pixels[JOINT_INDEX[name]] = pixels[JOINT_INDEX[mirror]] + np.array([20.0, 0.0])

    return Pose(world=world, pixels=pixels,
                confidence=np.full(NUM_JOINTS, confidence), timestamp=0.0)


class TestFraming:
    def test_well_framed_body_passes(self):
        r = check_framing(framed_pose(), 640, 480, REQUIRED)
        assert r.usable and r.coverage == 1.0
        assert "Framing looks good." in r.advice

    def test_missing_joint_is_named(self):
        pose = framed_pose()
        pose.confidence[JOINT_INDEX["left_knee"]] = 0.05
        r = check_framing(pose, 640, 480, REQUIRED)
        assert not r.usable
        assert "left_knee" in r.missing
        assert any("left knee" in a for a in r.advice)

    def test_no_person_says_so(self):
        r = check_framing(None, 640, 480, REQUIRED)
        assert not r.usable
        assert any("step into" in a.lower() for a in r.advice)

    def test_joint_at_the_edge_is_flagged(self):
        pose = framed_pose()
        pose.pixels[JOINT_INDEX["left_ankle"]] = (320.0, 477.0)
        r = check_framing(pose, 640, 480, REQUIRED)
        assert "left_ankle" in r.at_edge
        assert any("edge" in a for a in r.advice)

    def test_body_filling_the_frame_is_warned(self):
        r = check_framing(framed_pose(fill=0.99), 640, 480, REQUIRED)
        assert any("whole frame" in a for a in r.advice)

    def test_body_too_small_is_warned(self):
        r = check_framing(framed_pose(fill=0.2), 640, 480, REQUIRED)
        assert any("closer" in a for a in r.advice)

    def test_advice_says_which_way_to_move(self):
        """Being told something is wrong without what to change is useless."""
        pose = framed_pose()
        for name in ("left_knee", "left_ankle"):
            pose.confidence[JOINT_INDEX[name]] = 0.05
        # Remaining visible joints sit high in frame.
        for name in ("left_shoulder", "left_hip"):
            pose.pixels[JOINT_INDEX[name]] = (320.0, 60.0)
        r = check_framing(pose, 640, 480, REQUIRED)
        assert any("tilt it down" in a or "Back up" in a for a in r.advice)

    def test_partial_coverage_is_reported_numerically(self):
        pose = framed_pose()
        pose.confidence[JOINT_INDEX["left_knee"]] = 0.05
        pose.confidence[JOINT_INDEX["left_ankle"]] = 0.05
        r = check_framing(pose, 640, 480, REQUIRED)
        assert r.coverage == pytest.approx(0.5)
