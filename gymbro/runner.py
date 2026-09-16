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
from .engine.session import SetTracker
from .exercises.loader import get_exercise
from .pose.backends import create_backend
from .pose.cameras import CameraSource, SyncedPair, phone_camera
from .program.generator import DayPlan

GREEN, RED, YELLOW, WHITE = (80, 220, 100), (60, 60, 240), (40, 200, 240), (255, 255, 255)


def _speak(text: str) -> None:
    """Best-effort TTS. Never let a missing voice engine break a workout."""
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass


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


def run_set(tracker, backend, source, show_video, set_index, block) -> None:
    """Drive one set until the reps are done or the user quits."""
    last_rep_spoken = 0
    while not tracker.done:
        frame_data = source.read()
        if frame_data is None:
            continue
        frame, pair_frame, _skew = (
            frame_data if isinstance(frame_data, tuple) else (frame_data, None, 0.0)
        )
        image = cv2.flip(frame.image, 1)
        now = time.monotonic()

        pose = backend.estimate(image, now)
        feedback = tracker.update(pose, now)

        if feedback.rep_completed is not None and tracker.counter.count > last_rep_spoken:
            last_rep_spoken = tracker.counter.count
            _speak(str(last_rep_spoken))

        if show_video:
            cv2.imshow("gymbro", _draw_overlay(image, tracker, feedback, block, set_index))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt
            if key == ord("n"):
                return


def run_session(
    coach: Coach, plan: DayPlan, camera: int = 0, phone_url: str = "",
    backend: str = "mediapipe", show_video: bool = True,
) -> int:
    pose_backend = create_backend(backend)

    if phone_url:
        source = SyncedPair(CameraSource(camera, "laptop"), phone_camera(phone_url))
        print(f"Two-view mode: laptop {camera} + {phone_url}")
    else:
        source = CameraSource(camera, "laptop")

    if not source.open():
        print("Could not open the camera.")
        return 1

    session_id = coach.start_session(plan, backend)
    print(f"\n{plan.name} - {len(plan.blocks)} movements\n")

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
            _speak(exercise.name)

            for set_index in range(1, block.sets + 1):
                tracker = SetTracker(
                    exercise, target_reps=block.reps, side=block.side,
                    seeded_rom=seeded,
                )
                print(f"  Set {set_index}/{block.sets}...")
                run_set(tracker, pose_backend, source, show_video, set_index, block)

                result = tracker.finish()
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
        _speak("Session complete. Well done.")
    else:
        print(f"Session did not count: {verdict.reason}")
        if streak.debt:
            print(f"Missed sessions to clear: {streak.debt}")

    for note in summary["progression"]:
        print(f"  {note}")
    print("=" * 52 + "\n")
    return 0
