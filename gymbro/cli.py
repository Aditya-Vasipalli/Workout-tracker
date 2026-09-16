#!/usr/bin/env python3
"""gymbro command line.

    python -m gymbro.cli today            # what you're doing today
    python -m gymbro.cli train            # run today's session with the camera
    python -m gymbro.cli status           # streak, debt, recent history
    python -m gymbro.cli exercises        # browse the library
    python -m gymbro.cli export           # write your data out
    python -m gymbro.cli doctor           # check the camera/model setup
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from .coach import Coach, CoachConfig
from .exercises.loader import find, get_exercise, load_library
from .exercises.schema import Equipment
from .store.db import DEFAULT_DB, Store

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, YELLOW, RED, CYAN = "\033[32m", "\033[33m", "\033[31m", "\033[36m"


def _coach(args) -> Coach:
    store = Store(args.db)
    loads = tuple(float(x) for x in args.loads.split(",")) if args.loads else ()
    return Coach(store, CoachConfig(available_loads=loads, difficulty=args.difficulty))


# ----------------------------------------------------------------------- today


def cmd_today(args) -> int:
    coach = _coach(args)
    plan = coach.plan_for(date.today(), regenerate=args.regenerate)
    state = coach.streak_state()

    print(f"\n{BOLD}{plan.name}{RESET}  {DIM}{plan.plan_date}{RESET}")
    print(f"{DIM}~{plan.estimated_minutes} min | {plan.total_prescribed_reps()} reps prescribed{RESET}")

    if state.current:
        print(f"{GREEN}{state.current} session streak{RESET}"
              + (f"  {DIM}(best: {state.longest}){RESET}" if state.longest > state.current else ""))
    if plan.warning:
        print(f"\n{YELLOW}{plan.warning}{RESET}")

    print()
    for i, block in enumerate(plan.blocks, 1):
        ex = get_exercise(block.exercise_id)
        side = f" {CYAN}[{block.side}]{RESET}" if block.side else ""
        load = f" @ {block.load_kg:g}kg" if block.load_kg else ""
        unit = "s" if ex.is_isometric else ""
        print(f"  {i:2d}. {BOLD}{ex.name}{RESET}{side}")
        print(f"      {block.sets} x {block.reps}{unit}{load}   {DIM}rest {block.rest_s}s{RESET}")
        if ex.cues:
            print(f"      {DIM}{ex.cues[0]}{RESET}")
    print()
    return 0


# ----------------------------------------------------------------------- train


def cmd_train(args) -> int:
    try:
        from .runner import run_session
    except ImportError as exc:
        print(f"{RED}Camera dependencies missing: {exc}{RESET}")
        print("Install them with:  pip install -r requirements.txt")
        return 1

    coach = _coach(args)
    plan = coach.plan_for(date.today())
    return run_session(
        coach, plan,
        camera=args.camera,
        phone_url=args.phone,
        backend=args.backend,
        show_video=not args.headless,
        voice_enabled=not args.quiet,
        calibrate=args.calibrate,
        chattiness=args.chattiness,
    )


# ---------------------------------------------------------------------- status


def cmd_status(args) -> int:
    coach = _coach(args)
    st = coach.status()

    print(f"\n{BOLD}gymbro status{RESET}\n")
    print(f"  Current streak : {GREEN if st['streak'] else DIM}{st['streak']}{RESET}")
    print(f"  Longest streak : {st['longest']}")
    if st["debt"]:
        print(f"  {YELLOW}Missed sessions: {st['debt']}{RESET}")
    print(f"  Last session   : {st['last_session'] or 'never'}")

    if st["recent"]:
        print(f"\n{BOLD}Recent{RESET}")
        for s in st["recent"][:8]:
            mark = {"completed": f"{GREEN}done{RESET}",
                    "partial": f"{YELLOW}partial{RESET}",
                    "abandoned": f"{RED}abandoned{RESET}"}.get(s["status"], s["status"])
            print(f"  {s['plan_date']}  {s['plan_name']:22s} {mark}")
    print()
    return 0


# ------------------------------------------------------------------- exercises


def cmd_exercises(args) -> int:
    if args.show:
        ex = get_exercise(args.show)
        print(f"\n{BOLD}{ex.name}{RESET}  {DIM}({ex.id}){RESET}")
        print(f"  {ex.category} | {ex.difficulty} | "
              f"{', '.join(e.value for e in ex.equipment)}")
        print(f"  Targets: {', '.join(ex.primary_muscles)}")
        if ex.glute_emphasis:
            print(f"  Glute emphasis: {ex.glute_emphasis:.0%}")
        if ex.setup:
            print(f"\n  {BOLD}Setup{RESET}\n    {ex.setup}")
        if ex.cues:
            print(f"\n  {BOLD}Cues{RESET}")
            for c in ex.cues:
                print(f"    - {c}")
        print(f"\n  {BOLD}Camera{RESET}: {ex.preferred_view.value} view")
        if ex.rules:
            print(f"\n  {BOLD}What gets checked{RESET}")
            for r in ex.rules:
                print(f"    - {r.name}: {r.target} [{r.severity.value}]")
        print()
        return 0

    equipment = None
    if args.equipment:
        equipment = {Equipment(e.strip()) for e in args.equipment.split(",")}

    results = find(
        category=args.category, equipment=equipment,
        min_glute_emphasis=args.min_glute,
    )
    print(f"\n{len(results)} exercises\n")
    for ex in results:
        glute = f"{GREEN}glutes {ex.glute_emphasis:.0%}{RESET}" if ex.glute_emphasis >= 0.5 else ""
        print(f"  {ex.id:28s} {ex.name:32s} {DIM}{ex.difficulty:12s}{RESET} {glute}")
    print()
    return 0


# ---------------------------------------------------------------------- export


def cmd_export(args) -> int:
    from .store.export import export_all

    store = Store(args.db)
    paths = export_all(store, args.out)
    print(f"\nExported to {args.out}:")
    for p in paths:
        print(f"  {p.name}")
    print()
    return 0


# ----------------------------------------------------------------------- setup


def cmd_setup(args) -> int:
    """Live framing and camera-angle check. Use this before your first session."""
    try:
        import cv2
        from .pose.backends import create_backend
        from .pose.cameras import CameraSource
        from .pose.framing import check_framing
        from .pose.orientation import OrientationCalibrator, describe_tilt
    except ImportError as exc:
        print(f"{RED}Camera dependencies missing: {exc}{RESET}")
        return 1

    exercise = get_exercise(args.exercise)
    required = exercise.required_joints("left" if exercise.unilateral else None)

    print(f"\n{BOLD}Framing check: {exercise.name}{RESET}")
    print(f"  Wants a {exercise.preferred_view.value} view.")
    if exercise.setup:
        print(f"  {exercise.setup}")
    print(f"\n  Get into position. Press q when you're happy.\n")

    source = CameraSource(args.camera, "laptop")
    if not source.open():
        print(f"{RED}Could not open camera {args.camera}.{RESET}")
        return 1

    backend = create_backend(args.backend)
    calibrator = OrientationCalibrator(args.calibrate if args.calibrate != "off" else "floor")
    last_print = 0.0

    try:
        import time

        while True:
            frame = source.read()
            if frame is None:
                continue
            image = cv2.flip(frame.image, 1)
            h, w = image.shape[:2]
            pose = backend.estimate(image, time.monotonic())
            report = check_framing(pose, w, h, required)
            if pose is not None:
                calibrator.add(pose)

            if time.monotonic() - last_print > 1.5:
                last_print = time.monotonic()
                mark = f"{GREEN}ok{RESET}" if report.usable else f"{YELLOW}..{RESET}"
                print(f"  {mark} {report.coverage:.0%} of needed joints | " +
                      " ".join(report.advice))

            preview = image
            if pose is not None:
                for name in required:
                    from .pose.skeleton import JOINT_INDEX
                    x, y = pose.pixels[JOINT_INDEX[name]]
                    good = pose.confidence[JOINT_INDEX[name]] >= 0.4
                    cv2.circle(preview, (int(x), int(y)), 5,
                               (80, 220, 100) if good else (60, 60, 240), -1)
            cv2.putText(preview, f"{report.coverage:.0%} visible", (16, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (80, 220, 100) if report.usable else (40, 200, 240), 2)
            cv2.imshow("gymbro setup", preview)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        source.close()
        backend.close()
        cv2.destroyAllWindows()

    result = calibrator.result()
    if result is not None:
        print(f"\n  {describe_tilt(result)}")
    print()
    return 0


# ---------------------------------------------------------------------- doctor


def cmd_doctor(args) -> int:
    print(f"\n{BOLD}gymbro setup check{RESET}\n")
    ok = True

    for module, why in (
        ("cv2", "camera capture and display"),
        ("mediapipe", "3D pose estimation (the accurate path)"),
        ("numpy", "maths"),
        ("yaml", "exercise library"),
    ):
        try:
            __import__(module)
            print(f"  {GREEN}ok{RESET}    {module:12s} {DIM}{why}{RESET}")
        except ImportError:
            print(f"  {RED}MISSING{RESET} {module:12s} {DIM}{why}{RESET}")
            ok = False

    try:
        lib = load_library()
        print(f"  {GREEN}ok{RESET}    library      {DIM}{len(lib)} exercises{RESET}")
    except Exception as exc:
        print(f"  {RED}FAIL{RESET}  library      {exc}")
        ok = False

    try:
        from .pose.cameras import list_cameras
        cams = list_cameras()
        if cams:
            print(f"  {GREEN}ok{RESET}    cameras      {DIM}found at index {cams}{RESET}")
        else:
            print(f"  {YELLOW}none{RESET}  cameras      {DIM}no camera opened{RESET}")
    except Exception as exc:
        print(f"  {YELLOW}skip{RESET}  cameras      {DIM}{exc}{RESET}")

    print(f"\n  Database: {args.db}")
    print(f"\n{GREEN}Ready.{RESET}\n" if ok else f"\n{YELLOW}Install what's missing first.{RESET}\n")
    return 0 if ok else 1


# ------------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gymbro", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(DEFAULT_DB), help="database path")
    p.add_argument("--difficulty", default="intermediate",
                   choices=["beginner", "intermediate", "advanced"])
    p.add_argument("--loads", default="",
                   help="dumbbell weights you own, e.g. 5,10,15")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("today", help="show today's workout")
    t.add_argument("--regenerate", action="store_true", help="reroll today's plan")
    t.set_defaults(func=cmd_today)

    tr = sub.add_parser("train", help="run today's session with the camera")
    tr.add_argument("--camera", type=int, default=0, help="laptop camera index")
    tr.add_argument("--phone", default="", help="phone stream URL for a second view")
    tr.add_argument("--backend", default="mediapipe", choices=["mediapipe", "movenet"])
    tr.add_argument("--headless", action="store_true", help="no video window")
    tr.add_argument("--quiet", action="store_true", help="no spoken coaching")
    tr.add_argument("--chattiness", default="normal",
                    choices=["quiet", "normal", "pushy"],
                    help="how often it corrects you (default: normal)")
    tr.add_argument("--calibrate", default="floor",
                    choices=["floor", "standing", "off"],
                    help="how to find which way is up. 'floor' works lying "
                         "down; use it if you can't stand up in frame")
    tr.set_defaults(func=cmd_train)

    s = sub.add_parser("status", help="streak, debt and history")
    s.set_defaults(func=cmd_status)

    e = sub.add_parser("exercises", help="browse the library")
    e.add_argument("--category")
    e.add_argument("--equipment", help="comma separated, e.g. dumbbell,band")
    e.add_argument("--min-glute", type=float, default=0.0, dest="min_glute")
    e.add_argument("--show", help="detail for one exercise id")
    e.set_defaults(func=cmd_exercises)

    x = sub.add_parser("export", help="export your data")
    x.add_argument("--out", default="./gymbro_export")
    x.set_defaults(func=cmd_export)

    st = sub.add_parser("setup", help="live framing check for a cramped space")
    st.add_argument("--exercise", default="glute_bridge",
                    help="which exercise to frame for (default: glute_bridge)")
    st.add_argument("--camera", type=int, default=0)
    st.add_argument("--backend", default="mediapipe", choices=["mediapipe", "movenet"])
    st.add_argument("--calibrate", default="floor",
                    choices=["floor", "standing", "off"])
    st.set_defaults(func=cmd_setup)

    d = sub.add_parser("doctor", help="check the setup")
    d.set_defaults(func=cmd_doctor)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
