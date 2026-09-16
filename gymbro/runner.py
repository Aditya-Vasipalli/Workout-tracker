"""Live session runner: camera in, coached set out.

Imports cv2/mediapipe at module level deliberately -- the CLI catches the
ImportError and tells the user what to install, rather than failing halfway
through a workout.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .coach import Coach
from .coaching.policy import CoachingPolicy
from .coaching.voice import VoiceCoach, auto_speaker
from .engine.session import SetTracker
from .exercises.loader import get_exercise
from .pose.backends import create_backend
from .pose.cameras import CameraSource, SyncedPair, phone_camera
from .pose.orientation import Orientation, OrientationCalibrator, describe_tilt
from .program.generator import DayPlan

GREEN, RED, YELLOW, WHITE = (80, 220, 100), (60, 60, 240), (40, 200, 240), (255, 255, 255)


def calibrate_orientation(
    backend, source, voice, show_video: bool, method: str = "floor",
    seconds: float = 3.0,
) -> Orientation | None:
    """Work out which way is up, so a tilted laptop doesn't corrupt every angle.

    MediaPipe's world landmarks are camera-aligned, so without this every
    measurement against vertical reports the camera's pitch rather than the
    body's posture.
    """
    instruction = (
        "Lie on the mat with both feet flat and hold still."
        if method == "floor"
        else "Stand up straight and hold still."
    )
    print(f"\n  Calibrating camera angle. {instruction}")
    voice.say(instruction, key="calib", cooldown_s=0.0, ttl_s=6.0)

    calibrator = OrientationCalibrator(method, min_samples=int(seconds * 15))
    deadline = time.monotonic() + seconds + 4.0

    while time.monotonic() < deadline:
        frame_data = source.read()
        if frame_data is None:
            continue
        frame = frame_data[0] if isinstance(frame_data, tuple) else frame_data
        image = cv2.flip(frame.image, 1)
        pose = backend.estimate(image, time.monotonic())
        if pose is not None:
            calibrator.add(pose)

        if show_video:
            preview = image.copy()
            progress = min(1.0, len(calibrator._vectors) / max(calibrator.min_samples, 1))
            cv2.rectangle(preview, (20, 20), (20 + int(300 * progress), 44), (80, 220, 100), -1)
            cv2.rectangle(preview, (20, 20), (320, 44), (200, 200, 200), 1)
            cv2.putText(preview, instruction, (20, 74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow("gymbro", preview)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                return None

        if calibrator.ready and calibrator.stability > 0.97:
            break

    result = calibrator.result()
    if result is None:
        print("  Could not read the camera angle - continuing without correction.")
        print("  Angles will be less reliable if the camera is not level.")
        return None

    print(f"  {describe_tilt(result)}")
    if not result.reliable:
        print("  Calibration was unsteady, so it will not be applied. "
              "Re-run and hold still for a cleaner reading.")
        return None
    return result


def _draw_overlay(frame, tracker, feedback, block, set_index):
    ex = tracker.exercise
    h, w = frame.shape[:2]
    panel = np.zeros((h, 360, 3), dtype=np.uint8)

    def put(text, y, color=WHITE, scale=0.55, thick=1):
        cv2.putText(panel, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    put(ex.name[:28], 34, WHITE, 0.62, 2)
    side = f" [{tracker.side}]" if tracker.side else ""
    put(f"Set {set_index}{side}", 60, (170, 170, 170), 0.5)

    colour = GREEN if feedback.tracking_ok else RED
    put(f"{feedback.rep_count} / {tracker.target_reps}", 120, colour, 1.5, 3)
    put(f"phase: {feedback.phase}", 150, (170, 170, 170), 0.45)

    # Depth bar: how far into your own calibrated range this rep has gone.
    bar_y, bar_h = 180, 18
    cv2.rectangle(panel, (14, bar_y), (340, bar_y + bar_h), (60, 60, 60), -1)
    if np.isfinite(feedback.normalized):
        fill = int(326 * float(np.clip(feedback.normalized, 0, 1)))
        full = feedback.normalized >= ex.criteria.full_range_threshold
        cv2.rectangle(panel, (14, bar_y), (14 + fill, bar_y + bar_h),
                      GREEN if full else YELLOW, -1)
    target_x = 14 + int(326 * ex.criteria.full_range_threshold)
    cv2.line(panel, (target_x, bar_y - 4), (target_x, bar_y + bar_h + 4), WHITE, 2)
    put("depth", bar_y + 38, (170, 170, 170), 0.42)

    if feedback.message:
        words, line, y = feedback.message.split(), "", 250
        for word in words:
            if len(line) + len(word) > 32:
                put(line, y, YELLOW if feedback.tracking_ok else RED, 0.48)
                y, line = y + 22, word + " "
            else:
                line += word + " "
        put(line, y, YELLOW if feedback.tracking_ok else RED, 0.48)

    put("q quit   n skip set", h - 20, (120, 120, 120), 0.42)
    return np.hstack([frame, panel])


def run_set(tracker, backend, source, policy, show_video, set_index, block) -> bool:
    """Drive one set. Returns False if the user asked to skip or quit."""
    last_announced = 0
    last_rejection_seen = 0.0

    while not tracker.done:
        frame_data = source.read()
        if frame_data is None:
            continue
        frame = frame_data[0] if isinstance(frame_data, tuple) else frame_data
        image = cv2.flip(frame.image, 1)
        now = time.monotonic()

        pose = backend.estimate(image, now)
        feedback = tracker.update(pose, now)

        # Speech decisions all live in the policy; this loop never waits on audio.
        if not feedback.tracking_ok and feedback.message:
            policy.on_tracking_lost(feedback.message)
        else:
            policy.on_frame(feedback.form, feedback.tracking_ok)

        if feedback.rep_completed is not None and tracker.counter.count > last_announced:
            last_announced = tracker.counter.count
            policy.on_rep(feedback.rep_completed, last_announced, tracker.target_reps)

        rejection = tracker.counter.latest_rejection(since=last_rejection_seen)
        if rejection:
            policy.on_rejection(rejection)
            last_rejection_seen = now

        if show_video:
            cv2.imshow("gymbro", _draw_overlay(image, tracker, feedback, block, set_index))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt
            if key == ord("n"):
                return False
            if key == ord("m"):
                policy.voice.enabled = not policy.voice.enabled
                print(f"    voice {'on' if policy.voice.enabled else 'muted'}")

    return True


def run_session(
    coach: Coach, plan: DayPlan, camera: int = 0, phone_url: str = "",
    backend: str = "mediapipe", show_video: bool = True,
    voice_enabled: bool = True, calibrate: str = "floor",
    chattiness: str = "normal",
) -> int:
    pose_backend = create_backend(backend)

    from .engine.form import Severity
    cooldown_scale = {"quiet": 2.0, "normal": 1.0, "pushy": 0.5}[chattiness]
    voice = VoiceCoach(auto_speaker(), enabled=voice_enabled).start()
    policy = CoachingPolicy(voice, cooldowns={
        Severity.UNSAFE: 4.0 * cooldown_scale,
        Severity.FAULT: 7.0 * cooldown_scale,
        Severity.CUE: 10.0 * cooldown_scale,
    })

    if phone_url:
        source = SyncedPair(CameraSource(camera, "laptop"), phone_camera(phone_url))
        print(f"Two-view mode: laptop {camera} + {phone_url}")
    else:
        source = CameraSource(camera, "laptop")

    if not source.open():
        print("Could not open the camera.")
        voice.stop()
        return 1

    orientation = None
    if calibrate != "off":
        orientation = calibrate_orientation(
            pose_backend, source, voice, show_video, method=calibrate
        )

    session_id = coach.start_session(plan, backend)
    print(f"\n{plan.name} - {len(plan.blocks)} movements")
    print("  q quit | n skip set | m mute voice\n")

    try:
        for block in plan.blocks:
            exercise = get_exercise(block.exercise_id)
            seeded = coach.store.get_calibration(block.exercise_id, block.side or "")

            print(f"\n=== {exercise.name}"
                  f"{f' [{block.side}]' if block.side else ''} ===")
            if exercise.setup:
                print(f"  {exercise.setup}")
            print(f"  Camera: {exercise.preferred_view.value} view")
            if not seeded:
                print("  First time on this one - the first two reps calibrate "
                      "your range, so go through your full comfortable motion.")

            for set_index in range(1, block.sets + 1):
                tracker = SetTracker(
                    exercise, target_reps=block.reps, side=block.side,
                    seeded_rom=seeded, orientation=orientation,
                )
                policy.on_set_start(exercise.name, block.reps, block.side)
                print(f"  Set {set_index}/{block.sets}...")
                run_set(tracker, pose_backend, source, policy, show_video,
                        set_index, block)

                result = tracker.finish()
                policy.on_set_end(result.counted_reps, result.good_reps, block.reps)
                coach.record_set(session_id, result, set_index, block.load_kg)

                score = (f"{result.form_score:.0%}" if result.form_score is not None
                         else "not enough visibility to score")
                print(f"    {result.counted_reps} reps "
                      f"({result.good_reps} clean) | form {score}")
                for warning in result.warnings:
                    print(f"    ! {warning}")
                for cue in result.cues[:2]:
                    print(f"    -> {cue}")

                if set_index < block.sets:
                    print(f"    Rest {block.rest_s}s")
                    time.sleep(block.rest_s)

    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        voice.wait_idle(timeout=3.0)
        voice.stop()
        source.close()
        pose_backend.close()
        if show_video:
            cv2.destroyAllWindows()

    summary = coach.finish_session(session_id, __import__("datetime").date.today())
    verdict, streak = summary["verdict"], summary["streak"]

    print("\n" + "=" * 52)
    if verdict.completed:
        print(f"Session complete. {verdict.verified_reps}/{verdict.prescribed_reps} "
              f"verified reps.")
        print(f"Streak: {streak.current}")
    else:
        print(f"Session did not count: {verdict.reason}")
        if streak.debt:
            print(f"Missed sessions to clear: {streak.debt}")

    for note in summary["progression"]:
        print(f"  {note}")
    print("=" * 52 + "\n")
    return 0
