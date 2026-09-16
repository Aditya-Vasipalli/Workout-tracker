"""Pose estimation backends.

MediaPipe Pose (BlazePose GHUM) is the default because it emits
`pose_world_landmarks`: metric 3D coordinates in metres, rooted at the hip
midpoint. That is the single most important upgrade over the old tracker, which
computed angles from 2D pixel projections and therefore could not distinguish a
deep hip hinge from a shallow one performed closer to the camera.

All heavy imports are local so this module stays importable (and the rest of the
package testable) on a machine with no camera stack installed.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .skeleton import JOINT_INDEX, NUM_JOINTS, Pose


class PoseBackend(Protocol):
    name: str

    def estimate(self, frame: np.ndarray, timestamp: float) -> Pose | None: ...
    def close(self) -> None: ...


# MediaPipe's 33-landmark model -> our 17-joint canonical skeleton.
_MEDIAPIPE_MAP: dict[str, int] = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}


class MediaPipeBackend:
    """BlazePose GHUM via MediaPipe. Provides metric 3D world landmarks.

    Args:
        model_complexity: 0/1/2. Use 2 (heavy) on a discrete GPU -- it is
            markedly better on occluded and non-frontal poses, which is most of
            what a floor-based glute session looks like.
    """

    name = "mediapipe"

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        import mediapipe as mp

        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=False,  # we do our own, speed-adaptive smoothing
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def estimate(self, frame: np.ndarray, timestamp: float) -> Pose | None:
        import cv2

        height, width = frame.shape[:2]
        # MediaPipe handles its own letterboxing internally and takes RGB.
        result = self._pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not result.pose_world_landmarks or not result.pose_landmarks:
            return None

        world = np.zeros((NUM_JOINTS, 3))
        pixels = np.zeros((NUM_JOINTS, 2))
        confidence = np.zeros(NUM_JOINTS)

        wl = result.pose_world_landmarks.landmark
        pl = result.pose_landmarks.landmark

        for name, mp_idx in _MEDIAPIPE_MAP.items():
            i = JOINT_INDEX[name]
            w, p = wl[mp_idx], pl[mp_idx]
            # MediaPipe world landmarks are metres with Y pointing *down*.
            # Flip to a Y-up frame so "higher hips" means a larger Y.
            world[i] = (w.x, -w.y, w.z)
            pixels[i] = (p.x * width, p.y * height)
            confidence[i] = getattr(w, "visibility", 1.0)

        return Pose(
            world=world,
            pixels=pixels,
            confidence=confidence,
            timestamp=timestamp,
            source=self.name,
        )

    def close(self) -> None:
        self._pose.close()


class MoveNetBackend:
    """MoveNet Thunder via TFLite. 2D only -- kept for the legacy model file.

    This backend cannot populate real world landmarks. It fills `world` with
    pixel coordinates scaled to a nominal body height and a zero Z, so 3D
    consumers degrade predictably rather than silently reporting nonsense. Any
    form rule that needs depth should declare `requires_3d` and be skipped when
    running on this backend.
    """

    name = "movenet"
    provides_metric_3d = False

    _MAP: dict[str, int] = {
        "nose": 0,
        "left_shoulder": 5, "right_shoulder": 6,
        "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 9, "right_wrist": 10,
        "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14,
        "left_ankle": 15, "right_ankle": 16,
    }

    def __init__(self, model_path: str = "movenet_thunder.tflite", input_size: int = 256) -> None:
        try:
            import tflite_runtime.interpreter as tflite  # lighter, if present

            self._interpreter = tflite.Interpreter(model_path=model_path)
        except ImportError:
            import tensorflow as tf

            self._interpreter = tf.lite.Interpreter(model_path=model_path)

        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self.input_size = input_size

    def estimate(self, frame: np.ndarray, timestamp: float) -> Pose | None:
        from .geometry import letterbox

        height, width = frame.shape[:2]
        # Aspect-preserving. The original tracker's plain resize to a square
        # warped every angle it went on to measure.
        padded, tf_params = letterbox(frame, self.input_size)

        import cv2

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        self._interpreter.set_tensor(
            self._input_details[0]["index"], np.expand_dims(rgb, 0).astype(np.uint8)
        )
        self._interpreter.invoke()
        raw = self._interpreter.get_tensor(self._output_details[0]["index"])[0][0]

        world = np.zeros((NUM_JOINTS, 3))
        pixels = np.zeros((NUM_JOINTS, 2))
        confidence = np.zeros(NUM_JOINTS)

        for name, idx in self._MAP.items():
            i = JOINT_INDEX[name]
            y_norm, x_norm, score = raw[idx]
            # Undo the letterbox to get true source-image pixels.
            xy = tf_params.normalized_to_original(
                np.array([x_norm, y_norm]), width, height
            )
            pixels[i] = xy
            confidence[i] = float(score)

        # Pseudo-metric: scale pixels so shoulder-to-hip is a plausible 0.5m.
        torso_px = np.linalg.norm(
            pixels[JOINT_INDEX["left_shoulder"]] - pixels[JOINT_INDEX["left_hip"]]
        )
        scale = 0.5 / torso_px if torso_px > 1e-6 else 0.0
        hip_mid = (pixels[JOINT_INDEX["left_hip"]] + pixels[JOINT_INDEX["right_hip"]]) / 2
        world[:, 0] = (pixels[:, 0] - hip_mid[0]) * scale
        world[:, 1] = -(pixels[:, 1] - hip_mid[1]) * scale  # Y up
        world[:, 2] = 0.0

        return Pose(
            world=world,
            pixels=pixels,
            confidence=confidence,
            timestamp=timestamp,
            source=self.name,
            meta={"metric_3d": False},
        )

    def close(self) -> None:
        pass


def create_backend(name: str = "mediapipe", **kwargs):
    """Factory. Falls back to MoveNet only if explicitly asked."""
    if name == "mediapipe":
        return MediaPipeBackend(**kwargs)
    if name == "movenet":
        return MoveNetBackend(**kwargs)
    raise ValueError(f"unknown pose backend: {name!r}")
