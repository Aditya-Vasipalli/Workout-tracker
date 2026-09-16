import numpy as np
import pytest

from gymbro.pose.geometry import (
    angle_3d, letterbox_params, plane_normal, point_line_distance,
    signed_angle_about_axis, to_torso_frame, torso_frame,
)


class TestLetterbox:
    def test_preserves_aspect_ratio(self):
        tf = letterbox_params(640, 480, 256)
        # Uniform scale on both axes -- the bug in the old tracker was that
        # x scaled by 0.400 and y by 0.533.
        assert tf.scale == pytest.approx(256 / 640)
        assert tf.pad_x == 0
        assert tf.pad_y == pytest.approx((256 - 192) // 2)

    def test_roundtrip_recovers_original_pixels(self):
        tf = letterbox_params(640, 480, 256)
        orig = np.array([[320.0, 240.0], [100.0, 50.0], [639.0, 479.0]])
        padded = orig * tf.scale + np.array([tf.pad_x, tf.pad_y])
        assert np.allclose(tf.to_original(padded), orig)

    def test_normalized_roundtrip(self):
        tf = letterbox_params(1920, 1080, 256)
        orig = np.array([960.0, 540.0])
        px = orig * tf.scale + np.array([tf.pad_x, tf.pad_y])
        assert np.allclose(tf.normalized_to_original(px / 256, 1920, 1080), orig)

    def test_portrait_image(self):
        tf = letterbox_params(480, 640, 256)
        assert tf.scale == pytest.approx(256 / 640)
        assert tf.pad_y == 0 and tf.pad_x > 0

    def test_rejects_degenerate(self):
        with pytest.raises(ValueError):
            letterbox_params(0, 480, 256)

    def test_quantifies_old_distortion_bug(self):
        """Document the error the naive square resize introduced."""
        sx, sy = 256 / 640, 256 / 480           # the old code's implicit scales
        true_deg = 45.0
        limb = np.array([np.cos(np.radians(true_deg)), np.sin(np.radians(true_deg))])
        warped = np.degrees(np.arctan2(limb[1] * sy, limb[0] * sx))
        assert warped == pytest.approx(53.13, abs=0.1)
        # Letterboxing leaves the angle untouched.
        assert np.degrees(np.arctan2(limb[1] * sx, limb[0] * sx)) == pytest.approx(45.0)


class TestAngle3D:
    def test_right_angle(self):
        assert angle_3d([1, 0, 0], [0, 0, 0], [0, 1, 0]) == pytest.approx(90.0)

    def test_straight_line(self):
        assert angle_3d([1, 0, 0], [0, 0, 0], [-1, 0, 0]) == pytest.approx(180.0)

    def test_fully_closed(self):
        assert angle_3d([1, 0, 0], [0, 0, 0], [1, 0, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_stable_near_180(self):
        """arccos(dot) loses precision here; atan2 does not."""
        a = angle_3d([1, 0, 0], [0, 0, 0], [-1, 1e-7, 0])
        assert np.isfinite(a) and a == pytest.approx(180.0, abs=0.01)

    def test_stable_near_zero(self):
        a = angle_3d([1, 0, 0], [0, 0, 0], [1, 1e-7, 0])
        assert np.isfinite(a) and a == pytest.approx(0.0, abs=0.01)

    def test_degenerate_returns_nan(self):
        assert np.isnan(angle_3d([0, 0, 0], [0, 0, 0], [1, 0, 0]))

    def test_accepts_2d(self):
        assert angle_3d([1, 0], [0, 0], [0, 1]) == pytest.approx(90.0)

    def test_3d_angle_unaffected_by_rotation(self):
        """The whole point: rotating the subject must not change the angle."""
        a, b, c = np.array([1.0, 0, 0]), np.zeros(3), np.array([0.3, 0.9, 0])
        expected = angle_3d(a, b, c)
        for theta in np.linspace(0, np.pi, 7):
            R = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ])
            assert angle_3d(R @ a, R @ b, R @ c) == pytest.approx(expected, abs=1e-9)

    def test_2d_projection_degrades_with_rotation(self):
        """Contrast: the old 2D approach loses the angle as the subject turns."""
        a, b, c = np.array([1.0, 0, 0]), np.zeros(3), np.array([0.3, 0.9, 0])
        true_angle = angle_3d(a, b, c)
        theta = np.radians(50)
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ])
        # Drop Z, as a single camera does.
        projected = angle_3d((R @ a)[:2], (R @ b)[:2], (R @ c)[:2])
        assert abs(projected - true_angle) > 5.0


class TestSignedAngle:
    def test_sign_flips_with_direction(self):
        axis = np.array([0, 0, 1.0])
        pos = signed_angle_about_axis([1, 0, 0], [0, 0, 0], [0, 1, 0], axis)
        neg = signed_angle_about_axis([1, 0, 0], [0, 0, 0], [0, -1, 0], axis)
        assert pos == pytest.approx(90.0) and neg == pytest.approx(-90.0)

    def test_degenerate_axis(self):
        assert np.isnan(signed_angle_about_axis([1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]))


class TestPointLineDistance:
    def test_perpendicular_offset(self):
        assert point_line_distance([0, 1, 0], [-1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_on_the_line(self):
        assert point_line_distance([0.5, 0, 0], [0, 0, 0], [1, 0, 0]) == pytest.approx(0.0)

    def test_valgus_knee_measurable(self):
        """A knee caving 9cm inward off the hip-ankle line."""
        hip, ankle = np.array([0.1, 0.9, 0]), np.array([0.1, 0.0, 0])
        knee = np.array([0.01, 0.45, 0])
        assert point_line_distance(knee, hip, ankle) == pytest.approx(0.09, abs=1e-6)


class TestTorsoFrame:
    def test_orthonormal(self):
        f = torso_frame([-0.2, 1.4, 0], [0.2, 1.4, 0], [-0.15, 1.0, 0], [0.15, 1.0, 0])
        assert np.allclose(f @ f.T, np.eye(3), atol=1e-9)

    def test_invariant_under_yaw(self):
        """Body-local coordinates must not change when the subject turns."""
        ls, rs = np.array([-0.2, 1.4, 0]), np.array([0.2, 1.4, 0])
        lh, rh = np.array([-0.15, 1.0, 0]), np.array([0.15, 1.0, 0])
        probe = np.array([0.3, 0.7, 0.1])
        origin = (lh + rh) / 2
        base = to_torso_frame(probe, torso_frame(ls, rs, lh, rh), origin)

        theta = np.radians(65)
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ])
        r_ls, r_rs, r_lh, r_rh = (R @ p for p in (ls, rs, lh, rh))
        rotated = to_torso_frame(R @ probe, torso_frame(r_ls, r_rs, r_lh, r_rh), R @ origin)
        assert np.allclose(base, rotated, atol=1e-9)

    def test_degenerate_falls_back_to_identity(self):
        f = torso_frame([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])
        assert np.allclose(f, np.eye(3))
