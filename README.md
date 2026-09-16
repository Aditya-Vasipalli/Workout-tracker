# gymbro

A workout coach that prescribes your training, watches you do it, and only
gives you credit for reps it actually verified.

Built around dumbbells, resistance bands and a mat, with glute work as the
priority. 42 exercises: 17 glute-focused, 12 pilates, 13 dumbbell/band.

```bash
pip install -r requirements.txt
python -m gymbro.cli doctor      # check your setup
python -m gymbro.cli setup       # frame the camera (do this first)
python -m gymbro.cli today       # see today's session
python -m gymbro.cli train       # do it, with the camera watching
```

---

## Why the old tracker wasn't accurate

The previous version (`movenet_tracker.py`, kept as `README_old_tracker.md`)
had four problems, none of which were the pose model's fault. Worth reading if
you want to know what changed and why.

**1. The preprocessing warped every angle it measured.**
`preprocess_image` resized a 640×480 frame straight to 256×256. That compresses
x by 0.400 and y by 0.533 — a non-uniform squash. Every angle computed
afterwards was measured in a distorted space: a true 45° limb angle read as
53.1°, a true 30° read as 38°. Not noise, a consistent bias.
Fixed by aspect-preserving letterboxing (`gymbro/pose/geometry.py`). There's a
test that pins the old error at 53.13° so it can't come back.

**2. Two-dimensional angles can't measure three-dimensional movement.**
Angles came from projected pixel coordinates, so they were only correct if you
stood perfectly perpendicular to the lens. Rotate 30° and the measured angle
collapses toward the camera plane. For hip thrusts and RDLs, where depth *is*
the metric, this was fatal.
Fixed by using MediaPipe's `pose_world_landmarks` — real metric 3D coordinates
in metres — and doing all geometry in 3D. There's a test proving a 40° body
rotation changes the rep count and form score by nothing.

**3. The second camera did nothing.**
`display_dual_camera_feed` ran pose detection on both frames and then *picked
one*, discarding the other. No fusion, ever. You were paying for two cameras
and getting one camera's accuracy.
Now the metric-3D backend means one camera is genuinely enough. A phone as a
second view is optional (`--phone`), and when used, frames are matched by
timestamp with an enforced tolerance rather than assumed to be simultaneous.

**4. Rep counting was a state machine that couldn't work.**
`validate_progressive_movement` required movement through four ordered
thresholds. Its "skipped progression stage" guard compared stage gaps that were
always exactly 1 apart, so it could never fire — dead code. And stage 0's
threshold was frequently already satisfied at rest: for `glute_bridge`,
thresholds `[140,155,165,175]` meant stage 0 passed while you were lying
motionless on the floor.
Replaced with hysteresis on a signal normalised against **your own** calibrated
range, plus a refractory period. Fixed absolute thresholds were always going to
be wrong anyway — 165° of hip extension is a full lockout for one body and a
partial rep for another.

**Also:** form scores were invented numbers (`* 0.3`, `max(5, ...)`,
`form_score = 10`) that had never been validated against anything.

---

## How scoring works now

Every form rule is a named predicate with an explicit tolerance and a cue in
plain language. `gymbro exercises --show hip_thrust` lists exactly what gets
checked.

Rules are weighted by severity — a cue counts 1, a fault 2, something unsafe 4
— and the score is the fraction of weight passed.

Three properties worth knowing:

- **Rules are phase-scoped.** A lockout check only fires at the top of the rep.
  Checking it at the bottom, where failing is just what the bottom of a rep
  looks like, dropped clean sets to 0.88 before this was fixed.
- **Invisible joints are skipped, not guessed.** If the camera can't see your
  ankle, the rules needing it are reported as skipped with the reason.
- **Low coverage returns no score at all.** If under half the rules were
  checkable, you get "not enough visibility to score" instead of a
  confident-looking percentage derived from two visible joints.

Depth-dependent rules declare `requires_3d` and are skipped entirely on the 2D
MoveNet backend rather than silently producing nonsense.

---

## What "forces you" actually means

Software can't make you train. Here's what this does instead, stated plainly:

- **Only verified reps count.** A rep needs full range *and* correct tempo to
  be a good rep, and only good reps count toward completion (80% threshold).
  There's no self-report path and no skip key — the old `s` keypress that just
  incremented the counter is gone.
- **Load is gated on form, not just reps.** Two consecutive clean sessions
  raise the target; a session scored below 0.85 holds everything where it is.
  Adding weight on top of broken technique is how a form problem becomes an
  injury.
- **Missed sessions accrue as debt** whether or not you open the app, and debt
  makes tomorrow's session **shorter**, not longer. A backlog you can't clear
  is the fastest way to quit entirely. Debt caps at 3.
- **Streaks tolerate one missed day.** Your longest streak is never lost.

What this won't do is pretend app mechanics are willpower. If you want real
enforcement, the things that actually work are social or financial stakes.
These mechanisms make skipping *visible* and stop the program quietly
pretending you did sessions you didn't.

---

## Awkward camera angles

If the laptop sits on a table and you train on the floor underneath it, the
camera looks steeply down at you. That breaks more than it looks like it does,
so it is handled explicitly.

**The problem.** MediaPipe's world landmarks are *camera-aligned*: +Y is up in
the image, not up in the world. A camera pitched 60 degrees down delivers the
whole skeleton rotated by 60 degrees, and anything measured against vertical
then reports the camera's tilt rather than your posture. Measured before the
fix, `torso_upright` tracked camera pitch 1:1 -- a 60-degree camera moved the
reading by a full 60 degrees, and the height-based signals flipped sign.

**The fix.** A short calibration at the start of each session works out which
way is actually up, and every pose is rotated into a gravity-aligned frame
before anything measures it. Recovered camera tilt is accurate to under a
degree across 0-85 degrees of pitch in tests.

Two calibration methods, because a cramped space may not let you stand in frame:

```bash
python -m gymbro.cli train --calibrate floor      # default: lie flat, hold still
python -m gymbro.cli train --calibrate standing   # stand upright, hold still
python -m gymbro.cli train --calibrate off        # assume the camera is level
```

`floor` reads the plane through your shoulders and heels, which stay on the mat
even while your hips move, so it survives you bridging mid-calibration.

**What it does not fix.** Correcting the rotation does not recover information
the camera never had. Past roughly 45 degrees of tilt, depth-dependent
measurements degrade because the pose model is extrapolating a viewpoint it
saw little of in training. The session tells you the measured tilt and says so
when it is steep. Believe it.

Run `gymbro setup` before your first session: it shows the live view, marks
which joints it can and cannot see for a given exercise, and says which way to
move.

```bash
python -m gymbro.cli setup --exercise hip_thrust
```

---

## Spoken coaching

Live corrections while you lift, not a report afterwards.

```bash
python -m gymbro.cli train                       # coaching on
python -m gymbro.cli train --chattiness pushy    # corrects sooner, more often
python -m gymbro.cli train --chattiness quiet    # only the important things
python -m gymbro.cli train --quiet               # silent
```

Press `m` during a set to mute or unmute.

What it actually says, from a real simulated set where reps 3-6 were done with
the back arching past lockout:

```
[ 0.0s]  Glute Bridge. 9 reps.
[ 3.4s]  1
[ 7.1s]  2
[11.0s]  3
[11.0s]  Hold the squeeze at the top.
[14.7s]  4
[15.9s]  Tuck your pelvis under and drop your ribs - you're arching your
         back, not squeezing your glutes
[18.5s]  5
[22.2s]  6
[25.8s]  7
[25.8s]  That's it.
[33.6s]  Set complete. 5 of 9 were clean.
```

The design rules behind that, each of which took a bug to get right:

- **Speech never blocks the capture loop.** `pyttsx3.runAndWait()` blocks until
  the phrase finishes; calling it from the frame loop stalls pose estimation
  for the whole utterance. Speech runs on its own thread.
- **A fault must recur before it is called.** Single-frame faults are jitter.
  Faults are counted both over a frame window *and* per rep -- counting
  consecutive frames alone silently broke every phase-scoped rule, since a
  lockout check only runs at the top of a rep and could never accumulate.
- **One instruction at a time**, with a three-second gap. Three corrections at
  once means acting on none. Safety corrections may interrupt.
- **Cues say which way to move.** Sagging hips and piking hips need opposite
  advice, so the underlying measurement is signed -- `angle_3d` returns an
  interior angle capped at 180 and reports the same number for a 165-degree hip
  and a 195-degree one.
- **Ignoring a cue escalates it.** The same words a third time read as a stuck
  record.
- **Fixing something gets acknowledged**, judged per rep. Confirming on a run of
  clean frames congratulates you halfway up the rep you are still botching.
- **Invisible joints never produce cues.** A skipped rule is silent, not a
  confident instruction based on nothing.

### What the posture cues are actually measuring

Worth being precise, because the cue wording is more confident than the sensor:

- **"Tuck your pelvis under"** -- MediaPipe has no pelvis or spine landmarks, so
  true pelvic tilt is not observable. What is measured is the shoulder-hip-knee
  angle overshooting a straight line at the top of a bridge, which is what
  happens when you run out of hip extension and borrow the rest from your lower
  back. It catches the gross version, not a few degrees.
- **"Brace your core"** -- muscle activation cannot be seen. What is measured is
  the consequence: your hip line dropping below, or piking above, straight.
- **Symmetry, knee tracking, joint ranges** -- these are measured directly in 3D
  and are the most trustworthy of the lot.

---

## Fitness tracking

Data lives in SQLite at `~/.gymbro/gymbro.db`. Nothing depends on a third-party
API.

That's deliberate: **Google Fit can't be used here.** Google closed new
signups for the Fit APIs on 1 May 2024, and the whole family shuts down at the
end of 2026 ([migration
guide](https://developer.android.com/health-and-fitness/health-connect/migration/fit)).
Even writing the integration today, you couldn't get credentials. The
replacements are Health Connect (Android-only, on-device, needs a companion
app), the Google Health API (cloud, ex-Fitbit), and HealthKit on iOS.

`gymbro export` writes CSV, JSON and TCX. TCX imports directly into Strava and
Garmin Connect. If you later want Health Connect, the store has a clean
boundary to adapt.

---

## Camera setup

One camera is enough. `gymbro exercises --show <id>` tells you the view each
exercise wants.

- **Side view** for hinges and bridges — hip thrust, RDL, glute bridge.
  Camera at hip height, far enough back to see head to feet.
- **Front view** for anything where knees can cave — squats, lunges, clamshells.
- Even, diffuse light. Avoid a bright window behind you.

Optional second view from your phone over WiFi:

```bash
python -m gymbro.cli train --phone http://192.168.1.42:8080/video
```

Use any IP-webcam style app. Expect 100–300 ms of latency; frames are matched
by timestamp within a 50 ms tolerance and rejected outside it, so a drifting
stream degrades to single-camera rather than silently corrupting the pose.

---

## Calibration

The first two reps of a new exercise learn your range of motion, so go through
your full comfortable movement. After that the range is frozen and stored, and
"full range" means *your* full range.

If your mobility genuinely improves, re-run with `--regenerate` or clear the
`calibration` row for that exercise.

---

## Layout

```
gymbro/
  pose/        geometry, filtering, backends, camera sources
  engine/      rep counting, form rules, per-set tracking
  exercises/   schema, loader, and library/*.yaml
  program/     workout generation, streaks/debt, progression
  store/       SQLite and exports
  coach.py     orchestration
  runner.py    live camera loop
  cli.py       command line
```

Adding an exercise means adding a YAML record, not editing code. The engine
interprets the spec.

The core (`pose/geometry`, `pose/filters`, `engine/*`, `program/*`) is pure
numpy and has no camera or model dependency, which is why the rep and form
logic can be tested against synthetic poses:

```bash
python -m pytest tests/ -q      # 185 tests, no camera needed
```

---

## Known limitations

Stated because a coach you can't trust is worse than no coach.

- **Lumbar rounding is approximated.** MediaPipe has no spine landmarks, so
  `spine_neutral` uses shoulder-hip-knee as a proxy. It catches gross rounding
  on an RDL. It will not catch subtle positioning.
- **Load is what you tell it.** Nothing detects which dumbbell you picked up.
- **Occlusion is the main failure mode.** Floor work where your body hides your
  own joints is where tracking quality drops. The set result reports the
  fraction of frames tracked cleanly; below 60% you get a warning, and you
  should believe the warning.
- **Form rule tolerances are informed defaults, not clinically validated
  numbers.** They were chosen to flag the errors that show up in these
  movements; they haven't been checked against expert coach ratings. Treat cues
  as prompts to think about a position, not verdicts.
- **Nothing here has been tested on a real body yet.** Every test runs against
  synthetic poses. The geometry and rep logic are verified; the form tolerances
  are informed guesses that will need adjusting against real video.
- **MediaPipe runs on CPU** in its Python wheels. Fine in real time on your
  hardware, but your GPU is idle. See the note in `requirements.txt` for the
  ONNX route if you want it.
