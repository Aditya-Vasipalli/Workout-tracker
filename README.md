# AI Workout Tracker 🏋️‍♀️

✅ **Status**: Successfully tested with Python 3.10.1! All features working.

A comprehensive Python-based workout tracking system with **25+ exercises** that counts reps and analyzes form quality in real-time using your webcam.

## 🚀 Quick Start

```bash
# Run the main program
py -3.10 main.py
```

## 🎯 Three Versions Available

### 1. Enhanced Tracker (`enhanced_workout_tracker.py`) - **NEW!**
- **25+ exercises** including leg, core, upper body, and stretches
- **Real-time pose detection** using MediaPipe BlazePose
- **Advanced form analysis** with detailed scoring
- **Female voice text-to-speech** with intelligent feedback
- **Time-based exercises** (planks, holds) and rep-based
- **Requires Python 3.8-3.12** (MediaPipe limitation)

### 2. Original Tracker (`workout_tracker.py`)
- **Bicep curls and squats** with precise angle tracking
- **Real-time pose detection** using MediaPipe BlazePose
- **Form quality scoring** based on joint angles
- **Requires Python 3.8-3.12** (MediaPipe limitation)

### 3. Simple Tracker (`simple_workout_tracker.py`)
- **Motion detection** based rep counting
- **Works with any Python version** (including 3.13+)
- **Manual rep counter** for backup
- **Good for basic workout tracking**r 🏋️‍♂️

✅ **Status**: Successfully tested with Python 3.10.1! All features working.

A Python-based workout tracking system that counts reps and analyzes form quality in real-time using your webcam.

## Two Versions Available 🎯

### 1. Full Version (`workout_tracker.py`)
- **Real-time pose detection** using MediaPipe BlazePose
- **Precise angle calculations** for bicep curls and squats
- **Advanced form analysis** with joint tracking
- **Requires Python 3.8-3.12** (MediaPipe limitation)

### 2. Simple Version (`simple_workout_tracker.py`)
- **Motion detection** based rep counting
- **Works with any Python version** (including 3.13+)
- **Manual rep counter** for backup
- **Good for basic workout tracking**

## Features ✨

- **25+ Exercise Support** with intelligent pose detection
- **Automatic rep counting** with precise joint angle calculations
- **Advanced form quality scoring** (0-100%) based on:
  - Range of motion and proper technique
  - Rep timing and movement control
  - Joint stability and alignment
- **Female voice text-to-speech** with intelligent feedback
  - Only speaks when form is correct (60%+ score)
  - Counts reps aloud and provides encouragement
  - Exercise-specific guidance
- **Session logging** to binary files with detailed analytics
- **Time-based exercises** for holds and stretches
- **Interactive terminal interface** with progress tracking
- **Multiple workout modes** for different skill levels

## 💪 Supported Exercises (25+)

### 🦵 Leg Exercises
- **Donkey Kicks** - Quadruped leg raises
- **Fire Hydrants** - Lateral leg raises from hands and knees
- **Glute Bridges** - Hip thrusts from lying position
- **Hip Thrusts** - Elevated glute bridges
- **Plie Squats** - Wide stance squats
- **Lunges** - Forward/backward lunges
- **Courtesy Lunges** - Diagonal back lunges
- **Side Leg Raises** - Lateral leg lifts
- **Calf Raises** - Rising on toes
- **Regular Squats** - Hip-knee-ankle tracking

### 🧘 Core Exercises
- **V-Hold** - Seated V position hold ⏱️
- **Boat Hold (Center/Left/Right)** - Boat pose variations ⏱️
- **Planks** - Timed plank holds ⏱️
- **Crunches** - Abdominal crunches
- **Leg Raises** - Lying leg lifts
- **Flutter Kicks** - Alternating leg movements
- **Russian Twists** - Seated torso rotations

### 💪 Upper Body Exercises
- **Push-Ups** - Standard push-ups with form analysis
- **Shoulder Press** - Overhead pressing movements
- **Arnold Press** - Rotating shoulder press
- **Bent-Over Rows** - Pulling movements
- **Dumbbell Pullover** - Overhead pullover motion
- **Bicep Curls** - Arm curls with angle tracking

### 🎯 Specialized Exercises
- **Pilates Clamshell** - Side-lying leg opens

### 🧘‍♀️ Stretches (Time-Based ⏱️)
- **Butterfly Stretch** - Seated groin stretch
- **Cobra Stretch** - Prone back extension
- **Frog Stretch** - Wide knee hip stretch

*⏱️ = Time-based exercises (measured in seconds rather than reps)*

## Installation 🚀

### Check Compatibility First
```bash
python check_compatibility.py
```

### Install Packages
```bash
pip install opencv-python numpy pyttsx3
```

### For MediaPipe Support (Full Version)
MediaPipe requires Python 3.8-3.12:
```bash
# Only works with Python 3.8-3.12
pip install mediapipe
```

**Note**: If you have Python 3.13+, use the simple version instead.

## Usage 📝

### Main Launcher (Recommended)
```bash
py -3.10 main.py
```
Interactive menu with options for:
- Enhanced tracker (25+ exercises)
- Original tracker (bicep curls & squats)
- Simple tracker (motion detection)
- Workout log reading
- Compatibility checking

### Direct Usage

**Enhanced Tracker (25+ exercises):**
```bash
py -3.10 enhanced_workout_tracker.py
```

**Original Tracker:**
```bash
py -3.10 workout_tracker.py
```

**Simple Tracker (no MediaPipe):**
```bash
py -3.10 simple_workout_tracker.py
```

### Creating Workouts

The enhanced tracker supports custom workout creation:

```
Exercise examples:
• donkey_kicks - 3 sets x 15 reps
• planks - 3 sets x 30 seconds  
• push_ups - 3 sets x 12 reps
• v_hold - 3 sets x 20 seconds
• bicep_curl - 3 sets x 15 reps
```

### Controls During Exercise
- **'q'**: Quit camera/exercise
- **'s'**: Complete current set manually
- **Enter**: Ready for next set

## 🎯 Smart Features

### Intelligent Voice Feedback
- **Female voice** (first available female voice)
- **Only speaks on good form** (60%+ form score)
- **Rep counting** with encouraging phrases
- **Exercise-specific guidance**

### Advanced Form Analysis
- **Joint angle tracking** for precise movement detection
- **Range of motion validation** for each exercise
- **Stability checking** to detect shaking or rushed movements
- **Real-time form scoring** with detailed feedback

### Workout Analytics
- **Binary log files** with complete workout data
- **Form score tracking** per set and overall averages
- **Time stamps** for each set and rep
- **Progress visualization** in terminal

## 📋 Workout Plan Format

Enhanced tracker supports flexible workout creation:

```python
# Rep-based exercises
{"name": "bicep_curl", "sets": 3, "reps": 12}
{"name": "push_ups", "sets": 3, "reps": 15}
{"name": "donkey_kicks", "sets": 2, "reps": 20}

# Time-based exercises  
{"name": "planks", "sets": 3, "reps": 30}  # 30 seconds
{"name": "v_hold", "sets": 3, "reps": 15}  # 15 seconds
```

## 🏆 Example Workouts

### Beginner Full Body (15 minutes)
```python
workout_plan = [
    {"name": "squats", "sets": 2, "reps": 10},
    {"name": "push_ups", "sets": 2, "reps": 8},
    {"name": "planks", "sets": 2, "reps": 20},  # seconds
    {"name": "calf_raises", "sets": 2, "reps": 15}
]
```

### Advanced Core Focus (20 minutes)
```python
workout_plan = [
    {"name": "v_hold", "sets": 3, "reps": 30},      # seconds
    {"name": "russian_twists", "sets": 3, "reps": 20},
    {"name": "leg_raises", "sets": 3, "reps": 15},
    {"name": "flutter_kicks", "sets": 3, "reps": 25},
    {"name": "planks", "sets": 3, "reps": 45}       # seconds
]
```

### Lower Body Strength (25 minutes)
```python
workout_plan = [
    {"name": "squats", "sets": 3, "reps": 15},
    {"name": "lunges", "sets": 3, "reps": 12},
    {"name": "glute_bridges", "sets": 3, "reps": 18},
    {"name": "donkey_kicks", "sets": 3, "reps": 15},
    {"name": "calf_raises", "sets": 3, "reps": 20}
]
```

## 📁 File Structure

```
Workout/
├── main.py                          # 🚀 Main launcher (START HERE)
├── enhanced_workout_tracker.py      # 25+ exercise tracker
├── exercise_tracker.py              # Exercise detection algorithms  
├── workout_tracker.py               # Original bicep/squat tracker
├── simple_workout_tracker.py        # Motion detection version
├── check_compatibility.py           # System compatibility checker
├── examples.py                      # Usage examples and demos
├── demo.py                         # Interactive demo interface
├── requirements.txt                # Package dependencies
├── README.md                       # This documentation
└── workout_session_*.bin           # Generated workout logs
```

## 🚀 Getting Started (Quick Guide)

1. **Check your Python version** (3.10-3.12 recommended):
   ```bash
   py -3.10 check_compatibility.py
   ```

2. **Install packages** (if needed):
   ```bash
   py -3.10 -m pip install -r requirements.txt
   ```

3. **Start working out**:
   ```bash
   py -3.10 main.py
   ```

4. **Choose Enhanced Tracker** for full experience
5. **Create your custom workout** from 25+ exercises
6. **Position yourself** in front of camera
7. **Follow the voice guidance** and visual feedback
8. **Review your logs** to track progress

## 🎯 Tips for Best Results

- **Good lighting** - Well-lit room with clear background
- **Full body visible** - Make sure camera can see your whole body
- **Stable camera** - Mount or place camera at chest height
- **Clear space** - Enough room to move freely
- **Proper form** - Focus on technique over speed
- **Listen to voice** - Only counts reps with good form (60%+)

---

**Happy Training! 🏋️‍♀️💪**
    "form_violations": ["Rep too fast"]
}
```

Use the `read_log()` function to view human-readable summaries.

## Terminal Output Example 📺

```
🏋️‍♂️ Starting Bicep Curl

🏋️  Start Set 1 of Bicep Curl
Target: 12 reps
Rep 1 completed - Form: 88%
Rep 2 completed - Form: 92%
...
✅ Bicep Curl - Set 1: 12 reps complete  
🧠 Form Score: 90%

🎉 Workout complete!
Summary:
- Bicep Curl: 3 sets, 36 total reps, Avg Form: 87%
- Squat: 3 sets, 45 total reps, Avg Form: 92%
```

## Camera Setup Tips 📹

1. **Position yourself** so your full body (or at least the relevant joints) are visible
2. **Good lighting** helps MediaPipe track your pose better
3. **Stable camera** - avoid moving the camera during exercises
4. **Clear background** - plain walls work best
5. **Side view** often works better than front-facing for form analysis

## Troubleshooting 🔧

### Camera Issues
- Make sure no other apps are using your webcam
- Try different camera indices if you have multiple cameras
- Check lighting conditions

### MediaPipe Detection Issues  
- Ensure you're fully visible in the camera frame
- Try wearing fitted clothing for better pose detection
- Move closer/farther from camera for optimal detection

### Form Scoring Issues
- Make sure you're performing exercises in the camera's side view
- Check that the correct joints are visible (shoulders, elbows, wrists for bicep curls)
- Calibrate by checking the angle display on screen

## Extending the System 🛠️

To add new exercises:

1. **Add exercise config:**
   ```python
   "pushup": {
       "joints": ["shoulder", "elbow", "wrist"],
       "angle_threshold_low": 90,
       "angle_threshold_high": 170,
       "landmarks": [11, 13, 15]
   }
   ```

2. **Update the `process_rep()` method** with the new exercise logic

3. **Test with the new exercise** in your workout plan

## Files Structure 📁

```
Workout/
├── workout_tracker.py    # Main tracking system
├── examples.py          # Usage examples  
├── README.md           # This file
├── workout_session.bin # Generated log files
└── .venv/             # Python virtual environment
```

## Dependencies 📦

- **OpenCV** (`cv2`) - Camera capture and image processing
- **MediaPipe** (`mediapipe`) - Pose detection and landmark tracking
- **NumPy** (`numpy`) - Mathematical calculations
- **pyttsx3** - Text-to-speech functionality
- **pickle** - Binary file serialization

## Future Enhancements 🚀

- Support for more exercises (push-ups, pull-ups, etc.)
- Web dashboard for tracking progress over time
- Integration with fitness apps
- Video recording of workouts
- Multi-person detection
- Resistance band/weight detection
