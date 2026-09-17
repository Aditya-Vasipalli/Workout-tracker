# gymbro — context handoff

I'm continuing work started in a Claude Code cloud session. Everything below is
already built, committed, and pushed. Read this, then confirm you can run the
test suite before changing anything.

## Where the code is

Repo: `Aditya-Vasipalli/Workout-tracker`
Branch: `claude/workout-tracker-form-detection-f5wb5k` (NOT main — check it out)

```bash
git fetch origin
git checkout claude/workout-tracker-form-detection-f5wb5k
pip install -r requirements.txt      # needs Python <=3.12; mediapipe has no 3.13 wheel
python -m pytest tests/ -q           # expect: 185 passed
python -m gymbro.cli doctor
```

## What this is

An AI workout coach: prescribes a daily workout, watches me do it on camera,
counts only reps it verifies, and coaches my form out loud in real time.

Priorities, in order: **glutes** (my actual goal), pilates, dumbbell work.
Equipment I own: **dumbbells, resistance bands, a mat**. No bench, no rack.
42 exercises live in `gymbro/exercises/library/*.yaml` as data, not code.

## My physical setup — this constrains everything

I train in a **very cramped space**. The laptop sits on a **table** and I work
out **on the floor underneath it**. So the webcam looks steeply *down* at me,
close range, often cropping limbs. Optionally I can add a phone camera over
WiFi as a second view, but it must work well with just the laptop cam.

Laptop has an **RTX 4060**.

## Layout

```
gymbro/
  pose/        geometry, One Euro filtering, MediaPipe backend, orientation, framing, cameras
  engine/      repcounter (hysteresis), form (declarative rules), session (per-set tracking)
  exercises/   schema, loader, library/*.yaml
  program/     generator, accountability (streaks/debt/progression)
  coaching/    voice (non-blocking TTS), policy (what to say and when)
  store/       SQLite + CSV/JSON/TCX export
  coach.py     orchestration   runner.py  live camera loop   cli.py  commands
```

Core (`pose/geometry`, `pose/filters`, `engine/*`, `program/*`) is pure numpy
with no camera dependency — that's why 185 tests run with no hardware.
Synthetic pose rigs live in `tests/synthetic.py`.

## Design decisions that must NOT be silently undone

Each of these came from a measured bug. Reverting any reintroduces a real fault.

1. **Letterboxing, not square resize.** The old tracker resized 640x480 straight
   to 256x256, warping every angle (a true 45 deg read as 53.1). See
   `pose/geometry.py:letterbox`.
2. **3D angles on MediaPipe world landmarks**, never 2D pixel projections.
3. **Gravity alignment** (`pose/orientation.py`). MediaPipe world landmarks are
   *camera-aligned*, so my steep downward camera rotated the whole skeleton and
   `torso_upright` tracked camera pitch 1:1. A calibration step recovers "up"
   and rotates poses before anything measures them. The floor method uses
   **shoulders + heels** (both stay on the mat) — an earlier version used heels
   + hips and read 93 deg for a 55 deg camera, because hips are what lift.
4. **Signed hip measurement** (`engine/form.py:hip_line_angle`). `angle_3d`
   returns an interior angle capped at 180, so it reported the same number for a
   165 deg hip and a 195 deg one. Sagging vs piking hips need OPPOSITE cues.
5. **Rep counting by hysteresis** against my own calibrated range, not fixed
   thresholds. Partial and bounced reps are flagged, not silently counted.
6. **Phase-scoped form rules.** A lockout check only fires at the top of a rep.
7. **Fault persistence is per-rep AND per-frame-window.** Consecutive-frame
   counting made every phase-scoped rule uncueable (a rep peak is too short).
8. **Rep-weighted form scoring**, not frame-weighted. Frame weighting scored a
   set with 4 clearly faulty reps at 99%; rep weighting gives 85%.
9. **Form score returns None** when too few joints were visible — never a
   fabricated percentage.
10. **Speech never blocks the capture loop.** `pyttsx3.runAndWait()` blocks;
    it runs on a worker thread in `coaching/voice.py`.

## Commands

```bash
python -m gymbro.cli setup --exercise glute_bridge   # live framing check — run first
python -m gymbro.cli today                           # today's workout
python -m gymbro.cli train --loads 5,10,15           # do it, camera + voice
python -m gymbro.cli status                          # streak, debt, history
python -m gymbro.cli exercises --show hip_thrust     # what gets checked
```

`train` flags: `--calibrate floor|standing|off`, `--chattiness quiet|normal|pushy`,
`--quiet`, `--phone <url>`, `--headless`. Press `m` mid-set to mute.

## THE IMPORTANT PART — what's unverified

**Nothing has ever been tested on a real human body.** All 185 tests run against
synthetic poses. The geometry and rep logic are verified; the **form rule
tolerances are informed guesses** that have never been checked against real
video or expert coach ratings.

Also: past ~45 deg of camera tilt, depth-dependent measurements degrade because
BlazePose is extrapolating a viewpoint it barely saw in training. Correcting
the rotation recovers orientation, not information the camera never captured.

## What I want to do next

Live-test it under my table and tune. Specifically:

1. Run `gymbro.cli setup` and get framing workable in my cramped space.
2. Check the **camera tilt figure** printed during calibration is plausible.
3. Do a real set and find out:
   - Does rep counting match what I actually did?
   - Do cues fire when they should, and stay quiet when form is fine?
   - Are any cues plainly WRONG (worst case — e.g. telling me to lift hips
     that are already too high)?
4. Tune thresholds against what we observe. Key numbers:
   - `pelvic_control` neutral band 160–186 deg (`engine/form.py`)
   - `hip_sag` max 12 deg
   - `RepCounter` enter 0.65 / exit 0.30 / full_range 0.85 (per-exercise in YAML)
   - `CoachingPolicy` FAULT_THRESHOLD 10 frames, REP_FAULT_THRESHOLD 2 reps,
     CORRECTION_GAP_S 3.0
5. If MediaPipe's CPU inference is too slow or too inaccurate at my angle,
   consider ONNX Runtime + CUDA with RTMPose/ViTPose behind the existing
   `PoseBackend` protocol in `pose/backends.py`. Nothing downstream changes.

Start by running the test suite and `gymbro.cli doctor`, tell me what you see,
and don't change code until we've watched it fail on real video.
