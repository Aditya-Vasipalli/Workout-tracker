"""Synthetic pose sequences, so the engine can be tested without a camera."""

from __future__ import annotations

import numpy as np

from gymbro.pose.skeleton import JOINT_INDEX, NUM_JOINTS, Pose


def make_pose(joints: dict[str, tuple[float, float, float]], confidence: float = 0.95,
              timestamp: float = 0.0, metric_3d: bool = True) -> Pose:
    world = np.zeros((NUM_JOINTS, 3))
    conf = np.full(NUM_JOINTS, confidence)
    for name, xyz in joints.items():
        world[JOINT_INDEX[name]] = xyz
    return Pose(
        world=world, pixels=world[:, :2] * 500 + 320, confidence=conf,
        timestamp=timestamp, source="synthetic", meta={"metric_3d": metric_3d},
    )


def _rotate_about(v: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation of `v` about a unit `axis`."""
    t = np.radians(angle_deg)
    return (
        v * np.cos(t)
        + np.cross(axis, v) * np.sin(t)
        + axis * np.dot(axis, v) * (1 - np.cos(t))
    )


def glute_bridge_pose(
    hip_angle_deg: float,
    timestamp: float = 0.0,
    pelvis_tilt_deg: float = 0.0,
    yaw_deg: float = 0.0,
    knee_angle_deg: float = 92.0,
    confidence: float = 0.95,
) -> Pose:
    """A supine bridge with *exactly* the specified hip and knee angles.

    Joints are constructed by rotation rather than guessed offsets, so
    angle_3d(shoulder, hip, knee) == hip_angle_deg and
    angle_3d(hip, knee, ankle) == knee_angle_deg hold to machine precision.
    That matters: a rig whose knee angle drifts out of the anatomical range
    would make a correct form rule look like an engine bug.
    """
    thigh, torso, shank, half_width = 0.42, 0.50, 0.40, 0.17
    lateral = np.array([1.0, 0.0, 0.0])

    hip = np.zeros(3)
    shoulder = hip + np.array([0.0, -0.10, -torso])

    v_hip_shoulder = (shoulder - hip) / np.linalg.norm(shoulder - hip)
    v_hip_knee = _rotate_about(v_hip_shoulder, lateral, hip_angle_deg)
    knee = hip + v_hip_knee * thigh

    # Ankle placed so the knee angle is exactly knee_angle_deg, bending the
    # shin down toward the floor.
    v_knee_hip = -v_hip_knee
    v_knee_ankle = _rotate_about(v_knee_hip, lateral, -knee_angle_deg)
    ankle = knee + v_knee_ankle * shank

    tilt = np.radians(pelvis_tilt_deg)
    joints: dict[str, tuple[float, float, float]] = {}
    for side, sign in (("left", -1.0), ("right", 1.0)):
        offset = np.array([sign * half_width, 0.0, 0.0])
        lift = np.array([0.0, sign * np.sin(tilt) * half_width, 0.0])
        joints[f"{side}_shoulder"] = tuple(shoulder + offset)
        joints[f"{side}_hip"] = tuple(hip + offset + lift)
        joints[f"{side}_knee"] = tuple(knee + offset + lift)
        joints[f"{side}_ankle"] = tuple(ankle + offset + lift)
        joints[f"{side}_heel"] = tuple(ankle + offset + lift + np.array([0, -0.03, -0.05]))
        joints[f"{side}_foot_index"] = tuple(ankle + offset + lift + np.array([0, -0.02, 0.12]))
        joints[f"{side}_elbow"] = tuple(shoulder + offset + np.array([sign * 0.1, -0.05, 0.2]))
        joints[f"{side}_wrist"] = tuple(shoulder + offset + np.array([sign * 0.12, -0.08, 0.4]))
    joints["nose"] = tuple(shoulder + np.array([0.0, 0.08, -0.15]))

    pose = make_pose(joints, confidence=confidence, timestamp=timestamp)

    if yaw_deg:
        t = np.radians(yaw_deg)
        R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        pose.world = pose.world @ R.T
    return pose


def bridge_sequence(
    n_reps: int,
    bottom_deg: float = 138.0,
    top_deg: float = 176.0,
    fps: float = 30.0,
    rep_period: float = 3.2,
    hold_s: float = 1.2,
    noise_m: float = 0.0,
    seed: int = 0,
    **pose_kwargs,
):
    """Yield (pose, timestamp) for a run of bridges."""
    rng = np.random.default_rng(seed)
    t = 0.0
    up_frames = int(fps * rep_period * 0.35)
    hold_frames = int(fps * hold_s)
    down_frames = int(fps * rep_period * 0.45)

    for _ in range(n_reps):
        phases = (
            list(np.linspace(bottom_deg, top_deg, up_frames))
            + [top_deg] * hold_frames
            + list(np.linspace(top_deg, bottom_deg, down_frames))
        )
        for angle in phases:
            pose = glute_bridge_pose(angle, timestamp=t, **pose_kwargs)
            if noise_m:
                pose.world = pose.world + rng.normal(0, noise_m, pose.world.shape)
            yield pose, t
            t += 1.0 / fps


def supine_rest_pose(timestamp: float = 0.0, confidence: float = 0.95) -> Pose:
    """Someone lying flat on a mat, knees bent, feet planted.

    Shoulders, heels and the back of the head are all touching the floor, which
    is what the 'floor' orientation calibration asks the user to hold. The
    bridge rig places the ankle by rotation and so does not model floor contact;
    this does, and is what calibration accuracy should be measured against.

    Floor is the plane y = 0. Body thickness lifts the hips slightly above it.
    """
    j: dict[str, tuple[float, float, float]] = {}
    for side, sx in (("left", -0.17), ("right", 0.17)):
        j[f"{side}_shoulder"] = (sx, 0.00, -0.50)      # on the floor
        j[f"{side}_hip"] = (sx * 0.9, 0.09, 0.00)      # body thickness
        j[f"{side}_knee"] = (sx * 0.9, 0.42, 0.26)     # knees bent up
        j[f"{side}_ankle"] = (sx * 0.9, 0.07, 0.44)
        j[f"{side}_heel"] = (sx * 0.9, 0.00, 0.47)     # on the floor
        j[f"{side}_foot_index"] = (sx * 0.9, 0.06, 0.60)
        j[f"{side}_elbow"] = (sx * 1.3, 0.02, -0.28)
        j[f"{side}_wrist"] = (sx * 1.4, 0.02, -0.08)
    j["nose"] = (0.0, 0.10, -0.66)
    return make_pose(j, confidence=confidence, timestamp=timestamp)
