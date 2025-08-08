#!/usr/bin/env python3
"""
MoveNet Workout Tracker
Using TensorFlow Lite MoveNet for reliable pose detection
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import pyttsx3
import threading
import math
import json
import os
from datetime import datetime

class MoveNetWorkoutTracker:
    def __init__(self):
        # Initialize MoveNet
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.setup_movenet()
        
        # Initialize TTS
        self.tts_engine = None
        self.setup_tts()
        
        # Camera selection
        self.primary_camera_index = 0    # Laptop camera (adjustable)
        self.secondary_camera_index = 1  # Overhead/sky camera (fixed)
        self.dual_camera_mode = False    # Enable dual camera tracking
        self.camera_setup = None         # Store current camera configuration
        
        # Exercise tracking
        self.rep_count = 0
        self.state = 'down'
        self.last_rep_time = 0
        self.min_rep_time = 1.2
        self.angle_history = []
        self.confidence_threshold = 0.3
        self.last_form_score = None  # Track last form score for display
        
        # Workout session tracking
        self.current_session = {
            'start_time': None,
            'exercises': [],
            'total_duration': 0,
            'session_id': None
        }
        self.current_exercise_data = {
            'exercise_name': '',
            'exercise_type': '',
            'target_reps': 0,
            'completed_reps': 0,
            'sets_completed': 0,
            'target_sets': 0,
            'form_scores': [],
            'start_time': None,
            'end_time': None,
            'duration': 0
        }
        
        # Keypoint indices for MoveNet (17 keypoints)
        self.KEYPOINT_DICT = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        
        # Exercise definitions with comprehensive exercise library
        self.exercises = {
            # GLUTES
            'glute_bridge': {
                'name': 'Glute Bridge',
                'primary_muscle': 'Glutes',
                'equipment': 'Bodyweight/Dumbbell',
                'difficulty': 'Beginner',
                'type': 'Strength',
                'keypoints': ['left_shoulder', 'left_hip', 'left_knee'],
                'down_threshold': 140,
                'up_threshold': 170,
                'description': 'Lie down, lift hips up. Add pause at top for intensity',
                'camera_setup': {
                    'primary': 'side',          # Laptop camera for hip extension depth
                    'secondary': 'overhead',    # Sky camera for hip/knee alignment
                    'primary_tracking': True,
                    'form_validation': 'both'
                }
            },
            'hip_thrust': {
                'name': 'Hip Thrust',
                'primary_muscle': 'Glutes',
                'equipment': 'Bench + Dumbbell',
                'difficulty': 'Intermediate',
                'type': 'Strength',
                'keypoints': ['left_shoulder', 'left_hip', 'left_knee'],
                'down_threshold': 130,
                'up_threshold': 170,
                'description': 'Back on bench, feet shoulder width, spine neutral'
            },
            'frog_pump': {
                'name': 'Frog Pump',
                'primary_muscle': 'Glutes',
                'equipment': 'Bodyweight',
                'difficulty': 'Beginner',
                'type': 'Strength',
                'keypoints': ['left_knee', 'left_hip', 'right_knee'],  # Knee width tracking
                'down_threshold': 90,   # Knees closer (hips down)
                'up_threshold': 130,    # Knees wider (hips up)
                'description': 'Soles together, knees wide, hip thrust',
                'camera_setup': {
                    'primary': 'overhead',      # Sky camera for knee separation
                    'secondary': 'side',        # Laptop camera for hip thrust depth
                    'primary_tracking': True,   # Main tracking from overhead
                    'form_validation': 'both'   # Use both for complete form analysis
                },
                'form_cues': ['Keep soles together', 'Push knees out wide', 'Squeeze glutes at top']
            },
            'donkey_kick': {
                'name': 'Donkey Kick',
                'primary_muscle': 'Glutes',
                'equipment': 'Bodyweight',
                'difficulty': 'Beginner',
                'type': 'Strength',
                'bilateral': True,  # Exercise requires both sides
                'sides': {
                    'left': {
                        'keypoints': ['left_hip', 'left_knee', 'left_ankle'],
                        'name': 'Donkey Kick (Left)'
                    },
                    'right': {
                        'keypoints': ['right_hip', 'right_knee', 'right_ankle'],
                        'name': 'Donkey Kick (Right)'
                    }
                },
                'down_threshold': 90,
                'up_threshold': 160,
                'description': 'On hands and knees, kick back. Add ankle weights to progress',
                'camera_setup': {
                    'primary': 'overhead',      # Sky camera for leg position/alignment
                    'secondary': 'side',        # Laptop camera for kick height
                    'primary_tracking': True,   # Track from overhead
                    'form_validation': 'both'
                }
            },
            'fire_hydrant': {
                'name': 'Fire Hydrant',
                'primary_muscle': 'Glutes',
                'equipment': 'Bodyweight',
                'difficulty': 'Beginner',
                'type': 'Strength',
                'bilateral': True,  # Exercise requires both sides
                'sides': {
                    'left': {
                        'keypoints': ['left_hip', 'left_knee', 'left_ankle'],
                        'name': 'Fire Hydrant (Left)'
                    },
                    'right': {
                        'keypoints': ['right_hip', 'right_knee', 'right_ankle'],
                        'name': 'Fire Hydrant (Right)'
                    }
                },
                'down_threshold': 90,
                'up_threshold': 130,
                'description': 'Great for side glutes',
                'camera_setup': {
                    'primary': 'overhead',      # Sky camera for lateral movement
                    'secondary': 'side',        # Laptop camera for height validation
                    'primary_tracking': True,
                    'form_validation': 'both'
                }
            },
            'bulgarian_split_squat': {
                'name': 'Bulgarian Split Squat',
                'primary_muscle': 'Glutes/Quads',
                'equipment': 'Dumbbell + Bench',
                'difficulty': 'Intermediate',
                'type': 'Strength',
                'keypoints': ['left_hip', 'left_knee', 'left_ankle'],
                'down_threshold': 90,
                'up_threshold': 160,
                'description': 'Lean forward for glute focus',
                'camera_setup': {
                    'primary': 'side',          # Laptop camera for depth tracking
                    'secondary': 'overhead',    # Sky camera for balance/alignment
                    'primary_tracking': True,
                    'form_validation': 'both'
                }
            },
            'step_up': {
                'name': 'Step-Up',
                'primary_muscle': 'Glutes/Quads',
                'equipment': 'Dumbbell + Elevated Surface',
                'difficulty': 'Intermediate',
                'type': 'Strength',
                'keypoints': ['left_hip', 'left_knee', 'left_ankle'],
                'down_threshold': 90,
                'up_threshold': 160,
                'description': 'Full foot on platform',
                'camera_setup': {
                    'primary': 'side',          # Laptop camera for step height
                    'secondary': 'overhead',    # Sky camera for foot placement
                    'primary_tracking': True,
                    'form_validation': 'both'
                }
            },
            'glute_kickback': {
                'name': 'Glute Kickback',
                'primary_muscle': 'Glutes',
                'equipment': 'Bodyweight',
                'difficulty': 'Beginner',
                'type': 'Strength',
                'keypoints': ['left_hip', 'left_knee', 'left_ankle'],
                'down_threshold': 90,
                'up_threshold': 150,
                'description': 'Mind-muscle focus on extension'
            },
            'single_leg_glute_bridge': {
                'name': 'Single-leg Glute Bridge',
                'primary_muscle': 'Glutes',
                'equipment': 'Bodyweight',
                'difficulty': 'Intermediate',
                'type': 'Strength',
                'keypoints': ['left_shoulder', 'left_hip', 'left_knee'],
                'down_threshold': 140,
                'up_threshold': 170,
                'description': 'Great unilateral glute work'
            },
            'wall_sit_glute': {
                'name': 'Wall Sit (Glute Focus)',
                'primary_muscle': 'Glutes',
                'equipment': 'Bodyweight',
                'difficulty': 'Beginner',
                'type': 'Endurance',
                'keypoints': ['left_hip', 'left_knee', 'left_ankle'],
                'down_threshold': 85,
                'up_threshold': 95,
                'description': 'Press heels into floor'
            },
            
            # BICEPS / UPPER BODY
            'bicep_curl': {
                'name': 'Bicep Curl',
                'primary_muscle': 'Biceps',
                'equipment': 'Dumbbell',
                'difficulty': 'Beginner',
                'type': 'Strength',
                'keypoints': ['left_shoulder', 'left_elbow', 'left_wrist'],
                'down_threshold': 160,
                'up_threshold': 60,
                'description': 'Full range, no swinging'
            },
            'hammer_curl': {
                'name': 'Hammer Curl',
                'primary_muscle': 'Biceps/Brachialis',
                'equipment': 'Dumbbell',
                'difficulty': 'Intermediate',
                'type': 'Strength',
                'keypoints': ['left_shoulder', 'left_elbow', 'left_wrist'],
                'down_threshold': 160,
                'up_threshold': 60,
                'description': 'Neutral grip'
            },
            'concentration_curl': {
                'name': 'Concentration Curl',
                'primary_muscle': 'Biceps',
                'equipment': 'Dumbbell',
                'difficulty': 'Intermediate',
                'type': 'Strength',
                'keypoints': ['left_shoulder', 'left_elbow', 'left_wrist'],
                'down_threshold': 160,
                'up_threshold': 50,
                'description': 'Strict form seated'
            },
            'zottman_curl': {
                'name': 'Zottman Curl',
                'primary_muscle': 'Biceps/Forearms',
                'equipment': 'Dumbbell',
                'difficulty': 'Intermediate',
                'type': 'Strength',
                'keypoints': ['left_shoulder', 'left_elbow', 'left_wrist'],
                'down_threshold': 160,
                'up_threshold': 60,
                'description': 'Rotate on the way down'
            },
            'supinated_curl_to_press': {
                'name': 'Supinated Curl to Press',
                'primary_muscle': 'Biceps/Shoulders',
                'equipment': 'Dumbbell',
                'difficulty': 'Intermediate',
                'type': 'Hybrid',
                'keypoints': ['left_shoulder', 'left_elbow', 'left_wrist'],
                'down_threshold': 160,
                'up_threshold': 50,
                'description': 'Combo for compound work'
            },
            
            # CORE / ABS
            'plank': {
                'name': 'Plank',
                'primary_muscle': 'Core',
                'equipment': 'Bodyweight',
                'difficulty': 'Beginner',
                'type': 'Stability',
                'keypoints': ['left_shoulder', 'left_hip', 'left_ankle'],
                'down_threshold': 170,
                'up_threshold': 180,
                'description': 'Elbows under shoulders'
            },
            'side_plank': {
                'name': 'Side Plank',
                'primary_muscle': 'Obliques',
                'equipment': 'Bodyweight',
                'difficulty': 'Beginner',
                'type': 'Stability',
                'keypoints': ['left_shoulder', 'left_hip', 'left_ankle'],
                'down_threshold': 160,
                'up_threshold': 180,
                'description': 'Stack hips'
            },
            'leg_raise': {
                'name': 'Leg Raise',
                'primary_muscle': 'Lower Abs',
                'equipment': 'Bodyweight',
                'difficulty': 'Intermediate',
                'type': 'Strength',
                'keypoints': ['left_hip', 'left_knee', 'left_ankle'],
                'down_threshold': 30,
                'up_threshold': 90,
                'description': 'Use support under hips if needed'
            },
            'russian_twist': {
                'name': 'Russian Twist',
                'primary_muscle': 'Abs/Obliques',
                'equipment': 'Dumbbell',
                'difficulty': 'Beginner',
                'type': 'Strength',
                'keypoints': ['left_shoulder', 'right_shoulder', 'left_hip'],
                'down_threshold': 10,
                'up_threshold': 30,
                'description': 'Keep spine neutral'
            },
            
            # LEGS
            'goblet_squat': {
                'name': 'Goblet Squat',
                'primary_muscle': 'Quads/Glutes',
                'equipment': 'Dumbbell',
                'difficulty': 'Beginner',
                'type': 'Strength',
                'keypoints': ['left_hip', 'left_knee', 'left_ankle'],
                'down_threshold': 90,
                'up_threshold': 160,
                'description': 'Chest up, knees out'
            },
            'sumo_squat': {
                'name': 'Sumo Squat',
                'primary_muscle': 'Glutes/Inner Thighs',
                'equipment': 'Dumbbell',
                'difficulty': 'Beginner',
                'type': 'Strength',
                'keypoints': ['left_hip', 'left_knee', 'left_ankle'],
                'down_threshold': 90,
                'up_threshold': 160,
                'description': 'Toes out, wide stance'
            },
            'romanian_deadlift': {
                'name': 'Romanian Deadlift',
                'primary_muscle': 'Hamstrings/Glutes',
                'equipment': 'Dumbbell',
                'difficulty': 'Intermediate',
                'type': 'Strength',
                'keypoints': ['left_hip', 'left_knee', 'left_ankle'],
                'down_threshold': 90,
                'up_threshold': 170,
                'description': 'Soft knees, hinge hips'
            },
            'calf_raise': {
                'name': 'Calf Raise',
                'primary_muscle': 'Calves',
                'equipment': 'Bodyweight/Dumbbell',
                'difficulty': 'Beginner',
                'type': 'Strength',
                'keypoints': ['left_knee', 'left_ankle', 'left_ankle'],  # Special handling needed
                'down_threshold': 120,
                'up_threshold': 140,
                'description': 'Hold at top'
            },
            'dumbbell_lunge': {
                'name': 'Dumbbell Lunge',
                'primary_muscle': 'Quads/Hamstrings',
                'equipment': 'Dumbbell',
                'difficulty': 'Intermediate',
                'type': 'Strength',
                'keypoints': ['left_hip', 'left_knee', 'left_ankle'],
                'down_threshold': 90,
                'up_threshold': 160,
                'description': 'Controlled step'
            },
            
            # PILATES / MOBILITY
            'pilates_clamshell': {
                'name': 'Pilates Clamshell',
                'primary_muscle': 'Side Glutes',
                'equipment': 'Bodyweight',
                'difficulty': 'Beginner',
                'type': 'Strength',
                'keypoints': ['left_hip', 'left_knee', 'left_ankle'],
                'down_threshold': 80,
                'up_threshold': 120,
                'description': 'Add band if needed'
            },
            'bird_dog': {
                'name': 'Bird Dog',
                'primary_muscle': 'Core/Back',
                'equipment': 'Bodyweight',
                'difficulty': 'Beginner',
                'type': 'Stability',
                'keypoints': ['left_shoulder', 'left_hip', 'left_knee'],
                'down_threshold': 90,
                'up_threshold': 120,
                'description': 'Slow, precise'
            },
            
            # Original working exercises (keep for compatibility)
            'push_ups': {
                'name': 'Push-ups',
                'primary_muscle': 'Chest/Triceps',
                'equipment': 'Bodyweight',
                'difficulty': 'Intermediate',
                'type': 'Strength',
                'keypoints': ['left_shoulder', 'left_elbow', 'left_wrist'],
                'down_threshold': 90,
                'up_threshold': 160,
                'description': 'Keep body straight, full range of motion'
            },
            'bicep_curls': {
                'name': 'Bicep Curls', 
                'primary_muscle': 'Biceps',
                'equipment': 'Dumbbell',
                'difficulty': 'Beginner',
                'type': 'Strength',
                'keypoints': ['left_shoulder', 'left_elbow', 'left_wrist'],
                'down_threshold': 160,  # Straight arm
                'up_threshold': 60,     # Bent arm
                'description': 'Keep elbow close to body, full curl'
            },
            'squats': {
                'name': 'Squats',
                'primary_muscle': 'Quads/Glutes',
                'equipment': 'Bodyweight',
                'difficulty': 'Beginner',
                'type': 'Strength',
                'keypoints': ['left_hip', 'left_knee', 'left_ankle'], 
                'down_threshold': 90,   # Bent knee
                'up_threshold': 160,    # Straight leg
                'description': 'Squat down until thighs parallel to ground'
            },
            'shoulder_press': {
                'name': 'Shoulder Press',
                'primary_muscle': 'Shoulders',
                'equipment': 'Dumbbell',
                'difficulty': 'Intermediate',
                'type': 'Strength',
                'keypoints': ['left_shoulder', 'left_elbow', 'left_wrist'],
                'down_threshold': 90,   # Arms at shoulder level
                'up_threshold': 170,    # Arms extended overhead
                'description': 'Press weights overhead, keep core tight'
            }
        }
    
    def setup_movenet(self):
        """Download and setup MoveNet model"""
        try:
            print("📥 Downloading MoveNet model...")
            
            # Download MoveNet Thunder model
            model_url = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"
            model_path = "movenet_thunder.tflite"
            
            # Download model if not exists
            import urllib.request
            import os
            
            if not os.path.exists(model_path):
                print("⬇️ Downloading MoveNet model (this may take a moment)...")
                urllib.request.urlretrieve(model_url, model_path)
                print("✅ Model downloaded successfully")
            
            # Load the TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print("✅ MoveNet initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to setup MoveNet: {e}")
            print("Falling back to basic tracking...")
    
    def setup_tts(self):
        """Initialize text-to-speech with robust Windows implementation"""
        try:
            # Test if PowerShell TTS is available (more reliable on Windows)
            import subprocess
            test_command = '''
            Add-Type -AssemblyName System.Speech;
            $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;
            $synth.Speak("Test");
            '''
            
            result = subprocess.run([
                "powershell", "-Command", test_command
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                self.tts_method = 'powershell'
                self.tts_working = True
                print("✅ PowerShell TTS initialized")
            else:
                raise Exception("PowerShell TTS test failed")
                
        except Exception as e:
            # Fallback to pyttsx3
            try:
                self.tts_engine = pyttsx3.init(driverName='sapi5')
                voices = self.tts_engine.getProperty('voices')
                
                # Find female voice
                for voice in voices:
                    if 'zira' in voice.name.lower() or 'female' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                
                self.tts_engine.setProperty('rate', 150)
                self.tts_method = 'pyttsx3'
                self.tts_working = True
                print("✅ pyttsx3 TTS initialized (fallback)")
                
            except Exception as e2:
                print(f"⚠️ TTS setup failed: {e2}")
                self.tts_engine = None
                self.tts_method = 'none'
                self.tts_working = False
        
        self.tts_lock = threading.Lock()
    
    def speak(self, text):
        """
        Robust text-to-speech with fresh engine for each call
        """
        def powershell_tts():
            """Use Windows PowerShell TTS (most reliable)"""
            try:
                import subprocess
                ps_command = f'''
                Add-Type -AssemblyName System.Speech;
                $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;
                $synth.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female);
                $synth.Rate = 0;
                $synth.Speak("{text}");
                '''
                
                subprocess.run([
                    "powershell", "-Command", ps_command
                ], capture_output=True, text=True, timeout=10)
                
            except Exception as e:
                print(f"⚠️ PowerShell TTS error: {e}")
                try:
                    import winsound
                    winsound.Beep(800, 200)
                except:
                    pass
        
        print(f"� Speaking: {text}")
        
        if self.tts_working:
            thread = threading.Thread(target=powershell_tts)
            thread.daemon = True
            thread.start()
        else:
            print("⚠️ TTS not available - using visual feedback only")
            try:
                import winsound
                winsound.Beep(800, 100)  # Simple beep fallback
            except:
                pass
    
    def show_camera_placement_guide(self, exercise_type):
        """Show camera placement guide for specific exercise"""
        exercise = self.exercises.get(exercise_type, {})
        exercise_name = exercise.get('name', exercise_type)
        
        print(f"\n📹 CAMERA PLACEMENT GUIDE for {exercise_name}")
        print("=" * 50)
        
        # Exercise-specific guidance
        if exercise_type in ['glute_bridge', 'hip_thrust']:
            print("📱 SIDE VIEW SETUP:")
            print("   📍 Place phone on SIDE (left or right)")
            print("   📐 Phone should be at hip level")
            print("   📏 Distance: 3-4 feet away")
            print("   👁️  Should see: full hip, knee, ankle")
            print("   ✅ Good for: hip extension tracking")
            
        elif exercise_type == 'frog_pump':
            print("📱 **FRONT VIEW REQUIRED** for accurate tracking:")
            print("   📍 Place phone FACING your feet (head-to-toe view)")
            print("   📐 Phone at floor level, slight angle up")
            print("   📏 Distance: 4-5 feet away")
            print("   👁️  MUST see: both knees, hips, and knee separation")
            print("   ⚠️  CRITICAL: Camera must see both legs to track knee width!")
            print("   ✅ Good for: knee separation + hip thrust tracking")
            print("   🎯 FORM CUES: Keep soles together, push knees wide apart")
            
        elif exercise_type in ['leg_raise', 'russian_twist']:
            print("📱 SIDE VIEW SETUP:")
            print("   📍 Place phone on SIDE (left or right)")
            print("   📐 Phone at floor level or slightly elevated")
            print("   📏 Distance: 4-5 feet away")
            print("   👁️  Should see: full body from head to feet")
            print("   ✅ Good for: leg movement tracking")
            
        elif exercise_type in ['bicep_curls', 'shoulder_press']:
            print("📱 FRONT VIEW SETUP:")
            print("   📍 Place phone FACING you")
            print("   📐 Phone at chest/shoulder level")
            print("   📏 Distance: 4-6 feet away")
            print("   👁️  Should see: full upper body and arms")
            print("   ✅ Good for: arm movement tracking")
            
        elif exercise_type in ['squat', 'bulgarian_split_squat']:
            print("📱 SIDE VIEW SETUP:")
            print("   📍 Place phone on SIDE (left or right)")
            print("   📐 Phone at knee level")
            print("   📏 Distance: 5-6 feet away")
            print("   👁️  Should see: full body, especially legs")
            print("   ✅ Good for: knee/hip angle tracking")
            
        elif exercise_type in ['plank', 'push_up']:
            print("📱 SIDE VIEW SETUP:")
            print("   📍 Place phone on SIDE (left or right)")
            print("   📐 Phone at torso level")
            print("   📏 Distance: 4-5 feet away")
            print("   👁️  Should see: full body profile")
            print("   ✅ Good for: body alignment tracking")
            
        else:
            print("📱 GENERAL SETUP:")
            print("   📍 Place phone for best view of moving parts")
            print("   📐 Match phone level to exercise focus area")
            print("   📏 Distance: 4-6 feet away")
            print("   👁️  Ensure all key joints are visible")
        
        print("\n🎯 KEY TIPS:")
        print("   📱 Use phone LANDSCAPE mode")
        print("   💡 Good lighting on your body")
        print("   🔄 Test camera view before starting")
        print("   📐 Keep phone steady (use stand/prop)")
        print("   👥 Avoid background clutter")
        
        input("\n✅ Press Enter when camera is positioned...")

    def test_camera_view(self):
        """Live camera test to check positioning"""
        print("\n📹 CAMERA VIEW TEST")
        print("=" * 30)
        print("🎯 Check if all required keypoints are visible")
        print("📱 Adjust position until pose detection works well")
        print("❌ Press 'q' to quit test")
        
        cap = cv2.VideoCapture(self.primary_camera_index)
        if not cap.isOpened():
            print(f"❌ Could not open camera {self.primary_camera_index}")
            return
            
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("📹 Camera test started. Position yourself and check the view...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get pose landmarks
            keypoints = self.get_pose_landmarks(frame)
            
            if keypoints is not None:
                # Create extended display for camera test
                frame_height, frame_width = frame.shape[:2]
                text_area_width = 400
                extended_width = frame_width + text_area_width
                extended_frame = np.zeros((frame_height, extended_width, 3), dtype=np.uint8)
                
                # Draw pose on camera frame
                annotated_image = self.draw_keypoints(frame, keypoints)
                extended_frame[:, :frame_width] = annotated_image
                
                # Add status text in black area
                text_x = frame_width + 20
                cv2.putText(extended_frame, "CAMERA TEST", 
                           (text_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Count visible keypoints
                visible_points = sum(1 for kp in keypoints if kp[2] > self.confidence_threshold)
                cv2.putText(extended_frame, f"Visible Points:", 
                           (text_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(extended_frame, f"{visible_points}/17", 
                           (text_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if visible_points >= 10:
                    cv2.putText(extended_frame, "GOOD POSITION!", 
                               (text_x, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(extended_frame, "ADJUST POSITION", 
                               (text_x, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(extended_frame, "Press 'Q' to quit", 
                           (text_x, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                    
                cv2.imshow('Camera Test', extended_frame)
            else:
                # No pose detected - use extended layout too
                frame_height, frame_width = frame.shape[:2]
                text_area_width = 400
                extended_width = frame_width + text_area_width
                extended_frame = np.zeros((frame_height, extended_width, 3), dtype=np.uint8)
                extended_frame[:, :frame_width] = frame
                
                text_x = frame_width + 20
                cv2.putText(extended_frame, "NO POSE DETECTED", 
                           (text_x, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(extended_frame, "Position yourself", 
                           (text_x, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(extended_frame, "in camera frame", 
                           (text_x, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow('Camera Test', extended_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera test completed")

    def setup_dual_cameras(self, exercise_type):
        """Setup both cameras for dual tracking"""
        print("\n📹📹 DUAL CAMERA SETUP")
        print("=" * 50)
        
        exercise = self.exercises.get(exercise_type, {})
        camera_config = exercise.get('camera_setup', {})
        
        if not camera_config:
            print("❌ No dual camera config for this exercise")
            return False
        
        exercise_name = exercise.get('name', exercise_type)
        primary_angle = camera_config.get('primary', 'overhead')
        secondary_angle = camera_config.get('secondary', 'side')
        
        print(f"🎯 Exercise: {exercise_name}")
        print(f"📹 Primary Camera: {primary_angle.upper()} (main tracking)")
        print(f"📷 Secondary Camera: {secondary_angle.upper()} (form validation)")
        
        print(f"\n🎬 CAMERA INSTRUCTIONS:")
        
        # Sky camera (overhead) instructions
        if 'overhead' in [primary_angle, secondary_angle]:
            print("📹 SKY/OVERHEAD CAMERA (your fixed camera):")
            print("   ✅ Should already be positioned above workout area")
            print("   👁️  Should see: full body from above when lying down")
            print("   🎯 Good for: knee separation, leg alignment, arm positioning")
            print("   ⚠️  Make sure it captures the full mat area")
        
        # Laptop camera instructions  
        if primary_angle == 'side' or secondary_angle == 'side':
            print(f"\n💻 LAPTOP CAMERA - SIDE VIEW SETUP:")
            if exercise_type in ['glute_bridge', 'hip_thrust', 'bulgarian_split_squat']:
                print("   📍 Position laptop to your LEFT or RIGHT side")
                print("   📐 Camera at hip/knee level")
                print("   📏 Distance: 4-5 feet away")
                print("   👁️  Should see: full profile - shoulder to ankle")
                print("   🎯 Good for: hip extension depth, knee bend angle")
            elif exercise_type in ['squat', 'lunge']:
                print("   📍 Position laptop to your LEFT or RIGHT side")
                print("   📐 Camera at knee level")
                print("   📏 Distance: 5-6 feet away")
                print("   👁️  Should see: full body profile during movement")
                print("   🎯 Good for: squat depth, knee tracking")
            else:
                print("   📍 Position laptop to your LEFT or RIGHT side")
                print("   📐 Camera at exercise focus level")
                print("   📏 Distance: 4-6 feet away")
                print("   👁️  Should see: key joints clearly")
        
        if primary_angle == 'front' or secondary_angle == 'front':
            print(f"\n💻 LAPTOP CAMERA - FRONT VIEW SETUP:")
            print("   📍 Position laptop FACING you")
            print("   📐 Camera at chest/hip level")
            print("   📏 Distance: 4-6 feet away")
            print("   👁️  Should see: front view of exercise")
            print("   🎯 Good for: arm width, leg separation")
        
        print(f"\n🔧 TRACKING PRIORITY:")
        print(f"   🥇 Primary: {primary_angle.upper()} camera handles rep counting")
        print(f"   🥈 Secondary: {secondary_angle.upper()} camera validates form")
        print(f"   🎯 Both cameras contribute to form scoring")
        
        # Test both cameras
        print("\n🧪 CAMERA TEST SEQUENCE:")
        print("1. First we'll test the primary camera")
        print("2. Then we'll test the secondary camera") 
        print("3. Finally we'll run both together")
        
        input("\n✅ Press Enter when both cameras are positioned...")
        
        return True

    def select_camera_setup(self, exercise_type):
        """Select between single or dual camera setup"""
        exercise = self.exercises.get(exercise_type, {})
        camera_config = exercise.get('camera_setup')
        
        if camera_config:
            print(f"\n📹 Camera Setup Options for {exercise.get('name', exercise_type)}:")
            print("1. 🔥 DUAL CAMERA MODE (Recommended)")
            print("   - Sky camera + Laptop camera")
            print("   - Best form analysis")
            print("   - Complete movement tracking")
            print("2. 📱 SINGLE CAMERA MODE")
            print("   - Use laptop camera only")
            print("   - Basic tracking")
            
            choice = input("\nSelect mode (1-2): ").strip()
            
            if choice == '1':
                self.dual_camera_mode = True
                self.camera_setup = camera_config
                return self.setup_dual_cameras(exercise_type)
            else:
                self.dual_camera_mode = False
                self.select_camera()
                return True
        else:
            # No dual camera config, use single camera
            self.dual_camera_mode = False
            self.select_camera()
            return True

    def select_camera(self):
        """Let user select camera source"""
        print("\n📹 Camera Selection:")
        print("1. 💻 Laptop Camera Only")
        print("2. 📱 Phone Camera Only")
        print("3. 🎬 DUAL CAMERA SETUP (Laptop + Sky Camera)")
        print("4. 🔧 Custom Camera Index")
        
        try:
            choice = input("\nSelect camera (1-4): ").strip()
            
            if choice == '1':
                self.primary_camera_index = 0
                self.dual_camera_mode = False
                print("✅ Using laptop camera only")
            elif choice == '2':
                # Try common phone camera indices
                for i in [1, 2, 3]:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        cap.release()
                        self.primary_camera_index = i
                        self.dual_camera_mode = False
                        print(f"✅ Found phone camera at index {i}")
                        return
                print("❌ No phone camera found. Make sure Iriun is connected.")
                print("📱 Using laptop camera as fallback")
                self.primary_camera_index = 0
                self.dual_camera_mode = False
            elif choice == '3':
                self.primary_camera_index = 0      # Laptop camera
                self.secondary_camera_index = 1    # Sky camera
                self.dual_camera_mode = True
                print("✅ Using DUAL CAMERA setup!")
                print("📹 Primary: Laptop camera (adjustable)")
                print("🎥 Secondary: Sky camera (overhead)")
            elif choice == '4':
                idx = int(input("Enter camera index (0-5): "))
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.release()
                    self.primary_camera_index = idx
                    self.dual_camera_mode = False
                    print(f"✅ Using camera index {idx}")
                else:
                    print(f"❌ Camera {idx} not available. Using default.")
                    self.primary_camera_index = 0
                    self.dual_camera_mode = False
            else:
                print("❌ Invalid choice. Using laptop camera.")
                self.primary_camera_index = 0
                self.dual_camera_mode = False
                
        except ValueError:
            print("❌ Invalid input. Using laptop camera.")
            self.primary_camera_index = 0
            self.dual_camera_mode = False
        except Exception as e:
            print(f"❌ Camera selection error: {e}")
            self.primary_camera_index = 0
            self.dual_camera_mode = False

    def start_workout_session(self, session_name="Custom Workout"):
        """Start a new workout session"""
        self.current_session = {
            'session_name': session_name,
            'start_time': datetime.now(),
            'exercises': [],
            'total_duration': 0,
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        print(f"📊 Started workout session: {session_name}")

    def start_exercise_tracking(self, exercise_type, target_reps, target_sets=1):
        """Start tracking a new exercise"""
        exercise = self.exercises.get(exercise_type, {})
        self.current_exercise_data = {
            'exercise_name': exercise.get('name', exercise_type),
            'exercise_type': exercise_type,
            'target_reps': target_reps,
            'completed_reps': 0,
            'sets_completed': 0,
            'target_sets': target_sets,
            'form_scores': [],
            'start_time': datetime.now(),
            'end_time': None,
            'duration': 0,
            'primary_muscle': exercise.get('primary_muscle', 'Unknown'),
            'equipment': exercise.get('equipment', 'Unknown'),
            'difficulty': exercise.get('difficulty', 'Unknown')
        }

    def log_rep_completion(self, form_score):
        """Log a completed repetition"""
        self.current_exercise_data['completed_reps'] += 1
        self.current_exercise_data['form_scores'].append(form_score)
        self.last_form_score = form_score  # Store for display

    def complete_exercise_set(self):
        """Mark current set as completed"""
        self.current_exercise_data['sets_completed'] += 1

    def finish_exercise_tracking(self):
        """Finish tracking current exercise and add to session"""
        if self.current_exercise_data['start_time']:
            self.current_exercise_data['end_time'] = datetime.now()
            self.current_exercise_data['duration'] = (
                self.current_exercise_data['end_time'] - 
                self.current_exercise_data['start_time']
            ).total_seconds()
            
            # Calculate average form score
            if self.current_exercise_data['form_scores']:
                avg_form = sum(self.current_exercise_data['form_scores']) / len(self.current_exercise_data['form_scores'])
                self.current_exercise_data['avg_form_score'] = round(avg_form, 1)
            else:
                self.current_exercise_data['avg_form_score'] = 0
            
            # Add to session
            self.current_session['exercises'].append(self.current_exercise_data.copy())

    def save_workout_report(self):
        """Save workout session to JSON file"""
        if not self.current_session['exercises']:
            print("⚠️ No exercises to save")
            return None
            
        # Finish session
        self.current_session['end_time'] = datetime.now()
        self.current_session['total_duration'] = (
            self.current_session['end_time'] - 
            self.current_session['start_time']
        ).total_seconds()
        
        # Calculate session statistics
        total_reps = sum(ex['completed_reps'] for ex in self.current_session['exercises'])
        total_sets = sum(ex['sets_completed'] for ex in self.current_session['exercises'])
        all_form_scores = []
        for ex in self.current_session['exercises']:
            all_form_scores.extend(ex['form_scores'])
        
        avg_session_form = round(sum(all_form_scores) / len(all_form_scores), 1) if all_form_scores else 0
        
        self.current_session.update({
            'total_exercises': len(self.current_session['exercises']),
            'total_reps': total_reps,
            'total_sets': total_sets,
            'avg_session_form': avg_session_form,
            'muscles_worked': list(set(ex['primary_muscle'] for ex in self.current_session['exercises']))
        })
        
        # Convert datetime objects to ISO format strings for JSON compatibility
        session_copy = self.current_session.copy()
        session_copy['start_time'] = self.current_session['start_time'].isoformat()
        session_copy['end_time'] = self.current_session['end_time'].isoformat()
        
        # Convert exercise datetime objects
        for exercise in session_copy['exercises']:
            if exercise['start_time']:
                exercise['start_time'] = exercise['start_time'].isoformat()
            if exercise['end_time']:
                exercise['end_time'] = exercise['end_time'].isoformat()
        
        # Create report directory if it doesn't exist
        reports_dir = "workout_reports"
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        # Generate filename with current date
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(reports_dir, f"workout_{date_str}.json")
        
        # Save as JSON file
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_copy, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Workout report saved: {filename}")
            print(f"📊 Session Summary:")
            print(f"   • Duration: {self.current_session['total_duration']:.1f} seconds")
            print(f"   • Exercises: {self.current_session['total_exercises']}")
            print(f"   • Total Reps: {self.current_session['total_reps']}")
            print(f"   • Total Sets: {self.current_session['total_sets']}")
            print(f"   • Avg Form: {self.current_session['avg_session_form']}%")
            print(f"   • Muscles: {', '.join(self.current_session['muscles_worked'])}")
            
            return filename
            
        except Exception as e:
            print(f"❌ Error saving report: {e}")
            return None

    def load_workout_report(self, filename):
        """Load and display workout report from JSON file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Convert ISO format strings back to datetime objects for display
            start_time = datetime.fromisoformat(session_data['start_time'])
            
            print(f"\n📊 WORKOUT REPORT: {session_data['session_name']}")
            print("=" * 60)
            print(f"📅 Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏱️  Duration: {session_data['total_duration']:.1f} seconds ({session_data['total_duration']/60:.1f} minutes)")
            print(f"🏋️  Exercises: {session_data['total_exercises']}")
            print(f"🔢 Total Reps: {session_data['total_reps']}")
            print(f"📈 Total Sets: {session_data['total_sets']}")
            print(f"⭐ Avg Form: {session_data['avg_session_form']}%")
            print(f"💪 Muscles Worked: {', '.join(session_data['muscles_worked'])}")
            
            print(f"\n📋 EXERCISE BREAKDOWN:")
            print("-" * 60)
            for i, exercise in enumerate(session_data['exercises'], 1):
                print(f"{i}. {exercise['exercise_name']} ({exercise['primary_muscle']})")
                print(f"   • Completed: {exercise['completed_reps']}/{exercise['target_reps']} reps")
                print(f"   • Sets: {exercise['sets_completed']}/{exercise['target_sets']}")
                print(f"   • Avg Form: {exercise['avg_form_score']}%")
                print(f"   • Duration: {exercise['duration']:.1f}s")
                if exercise['form_scores']:
                    best_rep = max(exercise['form_scores'])
                    worst_rep = min(exercise['form_scores'])
                    print(f"   • Form Range: {worst_rep}% - {best_rep}%")
                print()
            
            return session_data
            
        except FileNotFoundError:
            print(f"❌ Report file not found: {filename}")
            return None
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in report file: {filename}")
            return None
        except Exception as e:
            print(f"❌ Error loading report: {e}")
            return None

    def list_workout_reports(self):
        """List all available workout reports"""
        reports_dir = "workout_reports"
        if not os.path.exists(reports_dir):
            print("📁 No workout reports found")
            return []
        
        reports = [f for f in os.listdir(reports_dir) if f.endswith('.json')]
        if not reports:
            print("📁 No workout reports found")
            return []
        
        reports.sort(reverse=True)  # Most recent first
        
        print(f"\n📊 AVAILABLE WORKOUT REPORTS ({len(reports)} found):")
        print("-" * 50)
        for i, report in enumerate(reports, 1):
            # Extract date from filename
            date_part = report.replace('workout_', '').replace('.json', '')
            try:
                date_obj = datetime.strptime(date_part, '%Y%m%d_%H%M%S')
                formatted_date = date_obj.strftime('%Y-%m-%d %H:%M:%S')
                print(f"{i}. {report} ({formatted_date})")
            except:
                print(f"{i}. {report}")
        
        return reports

    def preprocess_image(self, image):
        """Preprocess image for MoveNet"""
        # Resize to 256x256 (MoveNet Thunder input size)
        input_image = cv2.resize(image, (256, 256))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image, axis=0)
        input_image = tf.cast(input_image, dtype=tf.uint8)
        return input_image
    
    def get_pose_landmarks(self, image):
        """Get pose landmarks using MoveNet"""
        if self.interpreter is None:
            return None
        
        try:
            # Preprocess image
            input_image = self.preprocess_image(image)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_image.numpy())
            self.interpreter.invoke()
            
            # Get output
            keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Extract keypoints (y, x, score)
            keypoints = keypoints_with_scores[0][0]
            
            return keypoints
            
        except Exception as e:
            print(f"Pose detection error: {e}")
            return None
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        try:
            # Convert to numpy arrays
            a = np.array([p1[1], p1[0]])  # MoveNet returns (y,x)
            b = np.array([p2[1], p2[0]])
            c = np.array([p3[1], p3[0]])
            
            # Calculate vectors
            ba = a - b
            bc = c - b
            
            # Calculate angle
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            angle_degrees = np.degrees(angle)
            
            return angle_degrees
            
        except:
            return 0
    
    def track_exercise(self, exercise_type, keypoints):
        """Simple, reliable exercise tracking with bilateral support"""
        if exercise_type not in self.exercises:
            return False, 0, 0
        
        exercise = self.exercises[exercise_type]
        current_time = time.time()
        
        try:
            # Handle bilateral exercises
            if exercise.get('bilateral', False) and hasattr(self, 'current_side') and self.current_side:
                if self.current_side in ['left', 'right']:
                    # Use the specific side's keypoints
                    kp_names = exercise['sides'][self.current_side]['keypoints']
                elif self.current_side == 'both':
                    # For alternating sides, determine which side to track based on rep count
                    if not hasattr(self, 'alternating_side'):
                        self.alternating_side = 'left'  # Start with left
                    
                    # Switch sides every rep for alternating
                    current_alternating_side = 'left' if (self.rep_count % 2) == 0 else 'right'
                    kp_names = exercise['sides'][current_alternating_side]['keypoints']
                    
                    # Store which side we're currently tracking for feedback
                    self.current_tracking_side = current_alternating_side
                else:
                    kp_names = exercise.get('keypoints', [])
            else:
                # Regular exercise or fallback
                kp_names = exercise.get('keypoints', [])
                self.current_tracking_side = None
            
            if not kp_names:
                return False, 0, 0
            
            # Get keypoint indices
            kp_indices = [self.KEYPOINT_DICT[name] for name in kp_names]
            
            # Check if keypoints are detected with sufficient confidence
            points = []
            for idx in kp_indices:
                if keypoints[idx][2] < self.confidence_threshold:  # Low confidence
                    return False, 0, 0
                points.append([keypoints[idx][0], keypoints[idx][1]])
            
            # Special handling for exercises requiring specific measurements
            if exercise_type == 'frog_pump':
                # For frog pump: measure knee width (distance between knees)
                left_knee = points[0]   # left_knee
                hip = points[1]         # left_hip  
                right_knee = points[2]  # right_knee
                
                # Calculate knee separation distance
                knee_distance = math.sqrt((right_knee[0] - left_knee[0])**2 + (right_knee[1] - left_knee[1])**2)
                
                # Also calculate hip height relative to knees for thrust movement
                hip_height_left = abs(hip[1] - left_knee[1])  # Y difference (vertical)
                hip_height_right = abs(hip[1] - right_knee[1])
                avg_hip_height = (hip_height_left + hip_height_right) / 2
                
                # Use knee distance as primary angle, hip height as secondary
                angle = knee_distance * 2 + avg_hip_height  # Combined metric
                
            else:
                # Standard 3-point angle calculation
                angle = self.calculate_angle(points[0], points[1], points[2])
            
            # Add to history for smoothing
            self.angle_history.append(angle)
            if len(self.angle_history) > 5:
                self.angle_history.pop(0)
            
            # Use smoothed angle
            smoothed_angle = np.mean(self.angle_history)
            
            # Simple state machine
            rep_completed = False
            form_score = 0
            
            # Different logic for different exercises
            if exercise_type == 'bicep_curls':
                # For bicep curls: down = straight arm (large angle), up = bent arm (small angle)
                if self.state == 'down' and smoothed_angle < exercise['up_threshold'] + 15:
                    self.state = 'up'
                elif self.state == 'up' and smoothed_angle > exercise['down_threshold'] - 15:
                    if current_time - self.last_rep_time > self.min_rep_time:
                        rep_completed = True
                        self.last_rep_time = current_time
                        # Form score based on range achieved
                        form_score = min(100, max(60, int(100 - abs(smoothed_angle - exercise['down_threshold']))))
                    self.state = 'down'
            else:
                # For other exercises: down = small angle, up = large angle
                if self.state == 'down' and smoothed_angle > exercise['up_threshold'] - 15:
                    self.state = 'up'
                elif self.state == 'up' and smoothed_angle < exercise['down_threshold'] + 15:
                    if current_time - self.last_rep_time > self.min_rep_time:
                        rep_completed = True
                        self.last_rep_time = current_time
                        # Form score based on range achieved
                        form_score = min(100, max(60, int(100 - abs(smoothed_angle - exercise['down_threshold']))))
                    self.state = 'down'
            
            if rep_completed:
                self.rep_count += 1
            
            return rep_completed, form_score, smoothed_angle
            
        except Exception as e:
            print(f"Tracking error: {e}")
            return False, 0, 0
    
    def draw_keypoints(self, image, keypoints):
        """Draw keypoints on image"""
        height, width, _ = image.shape
        
        for i, keypoint in enumerate(keypoints):
            y, x, confidence = keypoint
            if confidence > self.confidence_threshold:
                cv2.circle(image, (int(x * width), int(y * height)), 4, (0, 255, 0), -1)
        
        return image
    
    def scale_keypoints_for_frame(self, keypoints, source_shape, target_shape):
        """Since MoveNet keypoints are normalized (0-1), they work on any frame size"""
        # MoveNet keypoints are already normalized, so we can use them directly
        # The draw_keypoints function will handle the scaling to pixel coordinates
        return keypoints
    
    def run_workout(self, exercise_type, target_reps=10, target_sets=1, set_number=1):
        """Run workout session with tracking"""
        if exercise_type not in self.exercises:
            print(f"❌ Exercise '{exercise_type}' not supported")
            return
        
        exercise = self.exercises[exercise_type]
        
        # Handle bilateral exercises (left/right sides)
        selected_side = None
        if exercise.get('bilateral', False):
            print(f"\n🔄 {exercise['name']} - Bilateral Exercise")
            print("Select which side to perform:")
            print("1. 👈 Left side")
            print("2. 👉 Right side") 
            print("3. 🔄 Both sides (alternating)")
            
            while True:
                try:
                    choice = input("Choose side (1-3): ").strip()
                    if choice == '1':
                        selected_side = 'left'
                        exercise_display_name = exercise['sides']['left']['name']
                        break
                    elif choice == '2':
                        selected_side = 'right'
                        exercise_display_name = exercise['sides']['right']['name']
                        break
                    elif choice == '3':
                        selected_side = 'both'
                        exercise_display_name = f"{exercise['name']} (Both Sides)"
                        break
                    else:
                        print("❌ Invalid choice. Please enter 1, 2, or 3.")
                except KeyboardInterrupt:
                    print("\n👋 Workout cancelled")
                    return
        else:
            exercise_display_name = exercise['name']
        
        # Store the selected side and current exercise configuration
        self.current_side = selected_side
        self.current_exercise_config = exercise
        
        # Store current workout parameters as instance variables
        self.current_target_sets = target_sets
        self.current_set_number = set_number
        
        # Start tracking if this is the first set
        if set_number == 1:
            self.start_exercise_tracking(exercise_type, target_reps, target_sets)
        
        print(f"\n🏋️‍♀️ Starting {exercise_display_name}")
        print(f"📋 {exercise['description']}")
        print(f"🎯 Target: {target_reps} reps")
        
        # Setup cameras (dual or single)
        if not self.select_camera_setup(exercise_type):
            print("❌ Camera setup failed")
            return
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
        
        print("🚀 GO!")
        self.speak("Go!")
        
        # Reset counters
        self.rep_count = 0
        self.state = 'down'
        self.last_rep_time = 0
        self.angle_history = []
        
        # Open camera(s)
        if self.dual_camera_mode:
            # Open both cameras
            primary_cap = cv2.VideoCapture(self.primary_camera_index)
            secondary_cap = cv2.VideoCapture(self.secondary_camera_index)
            
            if not primary_cap.isOpened():
                print(f"❌ Could not open primary camera {self.primary_camera_index}")
                return
            if not secondary_cap.isOpened():
                print(f"❌ Could not open secondary camera {self.secondary_camera_index}")
                print("🔄 Falling back to single camera mode")
                self.dual_camera_mode = False
                cap = primary_cap
            else:
                print("✅ Both cameras opened successfully")
                # Set camera properties
                for camera_cap in [primary_cap, secondary_cap]:
                    camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Smaller for dual display
                    camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera_cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            # Single camera mode
            cap = cv2.VideoCapture(self.primary_camera_index)
            if not cap.isOpened():
                print(f"❌ Could not open camera {self.primary_camera_index}")
                # Try fallback to default camera
                if self.primary_camera_index != 0:
                    print("🔄 Trying default camera...")
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        print("❌ No cameras available")
                        return
                else:
                    return
        
        print("📹 Camera started. Press 'q' to quit, 's' to skip to next rep")
        
        while self.rep_count < target_reps:
            if self.dual_camera_mode:
                # Read from both cameras
                ret1, frame1 = primary_cap.read()
                ret2, frame2 = secondary_cap.read()
                
                if not ret1 or not ret2:
                    print("📹 Camera feed lost")
                    break
                
                # Flip frames
                frame1 = cv2.flip(frame1, 1)
                frame2 = cv2.flip(frame2, 1)
                
                # Create dual camera display (will handle pose detection internally)
                self.display_dual_camera_feed(frame1, frame2, None, exercise, target_reps, exercise_type)
                
            else:
                # Single camera mode
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame
                frame = cv2.flip(frame, 1)
                
                # Get pose
                keypoints = self.get_pose_landmarks(frame)
                
                # Create extended display with text area
                self.display_single_camera_feed(frame, keypoints, exercise, target_reps, exercise_type)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.rep_count += 1
                print(f"⏭️ Skipped to rep {self.rep_count}")
                self.speak(str(self.rep_count))
        
        # Cleanup cameras
        if self.dual_camera_mode:
            primary_cap.release()
            secondary_cap.release()
        else:
            cap.release()
        cv2.destroyAllWindows()
        
        # Mark set as completed if target reps reached
        if self.rep_count >= target_reps:
            print(f"\n🎉 Set complete! {self.rep_count} reps finished!")
            self.speak("Set complete! Great job!")
            self.complete_exercise_set()
        else:
            print(f"\n⏹️ Set stopped at {self.rep_count} reps")
        
        # Finish exercise tracking if this was the last set
        if self.current_set_number >= self.current_target_sets:
            self.finish_exercise_tracking()

    def display_single_camera_feed(self, frame, keypoints, exercise, target_reps, exercise_type):
        """Display single camera feed with extended text area"""
        frame_height, frame_width = frame.shape[:2]
        text_area_width = 400
        extended_width = frame_width + text_area_width
        
        # Create extended frame with black background for text area
        extended_frame = np.zeros((frame_height, extended_width, 3), dtype=np.uint8)
        extended_frame[:, :frame_width] = frame
        
        if keypoints is not None:
            # Draw keypoints on camera frame
            extended_frame[:, :frame_width] = self.draw_keypoints(frame, keypoints)
            
            # Track exercise
            rep_completed, form_score, current_angle = self.track_exercise(exercise_type, keypoints)
            
            if rep_completed:
                # Add side information for bilateral exercises
                side_info = ""
                if hasattr(self, 'current_tracking_side') and self.current_tracking_side:
                    side_info = f" ({self.current_tracking_side.upper()} side)"
                
                print(f"✅ Rep {self.rep_count} completed! Form: {form_score}%{side_info}")
                self.speak(str(self.rep_count))
                self.log_rep_completion(form_score)
            
            # Display info in text area
            self.add_text_overlay(extended_frame, frame_width, exercise, target_reps, current_angle)
        else:
            # No pose detected
            text_x = frame_width + 20
            cv2.putText(extended_frame, "⚠ POSITION YOURSELF", 
                       (text_x, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(extended_frame, "IN CAMERA FRAME", 
                       (text_x, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('MoveNet Workout Tracker', extended_frame)

    def display_dual_camera_feed(self, frame1, frame2, keypoints, exercise, target_reps, exercise_type):
        """Display dual camera feed side by side with independent pose detection"""
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # Resize frames to same height
        target_height = 400
        new_w1 = int(w1 * target_height / h1)
        new_w2 = int(w2 * target_height / h2)
        
        frame1_resized = cv2.resize(frame1, (new_w1, target_height))
        frame2_resized = cv2.resize(frame2, (new_w2, target_height))
        
        # Create combined display
        total_width = new_w1 + new_w2 + 300  # Extra space for text
        combined_frame = np.zeros((target_height, total_width, 3), dtype=np.uint8)
        
        # Get pose from both cameras independently
        keypoints1 = self.get_pose_landmarks(frame1_resized)
        keypoints2 = self.get_pose_landmarks(frame2_resized)
        
        # Draw pose keypoints on both cameras
        frame1_with_pose = self.draw_keypoints(frame1_resized, keypoints1) if keypoints1 is not None else frame1_resized
        frame2_with_pose = self.draw_keypoints(frame2_resized, keypoints2) if keypoints2 is not None else frame2_resized
        
        # Place frames in combined display
        combined_frame[:, :new_w1] = frame1_with_pose
        combined_frame[:, new_w1:new_w1+new_w2] = frame2_with_pose
        
        # Intelligent camera selection based on exercise type
        exercise_config = self.exercises.get(exercise_type, {})
        camera_setup = exercise_config.get('camera_setup', {})
        
        # Determine which camera to use for tracking based on exercise configuration
        if camera_setup.get('primary') == 'overhead' and keypoints2 is not None:
            # Use overhead (secondary) camera for tracking
            tracking_keypoints = keypoints2
            primary_cam_for_tracking = False
        elif keypoints1 is not None:
            # Use laptop (primary) camera for tracking
            tracking_keypoints = keypoints1
            primary_cam_for_tracking = True
        elif keypoints2 is not None:
            # Fallback to secondary camera if primary fails
            tracking_keypoints = keypoints2
            primary_cam_for_tracking = False
        else:
            tracking_keypoints = None
            primary_cam_for_tracking = True
        
        if tracking_keypoints is not None:
            # Track exercise using the selected keypoints
            rep_completed, form_score, current_angle = self.track_exercise(exercise_type, tracking_keypoints)
            
            if rep_completed:
                cam_source = "OVERHEAD" if not primary_cam_for_tracking else "LAPTOP"
                
                # Add side information for bilateral exercises
                side_info = ""
                if hasattr(self, 'current_tracking_side') and self.current_tracking_side:
                    side_info = f" - {self.current_tracking_side.upper()} side"
                
                print(f"✅ Rep {self.rep_count} completed! Form: {form_score}% (via {cam_source} camera{side_info})")
                self.speak(str(self.rep_count))
                self.log_rep_completion(form_score)
            
            # Add text overlay
            self.add_text_overlay(combined_frame, new_w1 + new_w2, exercise, target_reps, current_angle)
            
            # Add camera labels with pose detection and tracking status
            primary_status = "✅ POSE" if keypoints1 is not None else "❌ NO POSE"
            secondary_status = "✅ POSE" if keypoints2 is not None else "❌ NO POSE"
            
            if keypoints1 is not None or keypoints2 is not None:
                tracking_status = " 🎯 TRACKING" if not primary_cam_for_tracking else " 🎯 TRACKING"
                if not primary_cam_for_tracking:
                    secondary_status += tracking_status
                else:
                    primary_status += tracking_status
            
            cv2.putText(combined_frame, f"PRIMARY {primary_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(combined_frame, f"SECONDARY {secondary_status}", (new_w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        cv2.imshow('Dual Camera Workout Tracker', combined_frame)

    def add_text_overlay(self, frame, text_start_x, exercise, target_reps, current_angle):
        """Add text information overlay"""
        text_x = text_start_x + 20
        
        # Exercise information
        cv2.putText(frame, f"Exercise:", 
                   (text_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{exercise['name']}", 
                   (text_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Rep counter
        cv2.putText(frame, f"Reps:", 
                   (text_x, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{self.rep_count}/{target_reps}", 
                   (text_x, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Current angle/measurement
        cv2.putText(frame, f"Measurement:", 
                   (text_x, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{current_angle:.1f}", 
                   (text_x, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Movement state
        cv2.putText(frame, f"State:", 
                   (text_x, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{self.state}", 
                   (text_x, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Form score
        if hasattr(self, 'last_form_score') and self.last_form_score is not None:
            cv2.putText(frame, f"Form Score:", 
                       (text_x, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            color = (0, 255, 0) if self.last_form_score >= 80 else (0, 255, 255) if self.last_form_score >= 60 else (0, 0, 255)
            cv2.putText(frame, f"{self.last_form_score}%", 
                       (text_x, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Current tracking side for bilateral exercises
        if hasattr(self, 'current_tracking_side') and self.current_tracking_side:
            cv2.putText(frame, f"Tracking Side:", 
                       (text_x, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            side_color = (255, 0, 255) if self.current_tracking_side == 'left' else (0, 255, 255)
            cv2.putText(frame, f"{self.current_tracking_side.upper()}", 
                       (text_x, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, side_color, 2)
    
    def load_workout_from_json(self, filename):
        """Load workout from JSON file"""
        try:
            with open(filename, 'r') as f:
                workout_data = json.load(f)
            return workout_data
        except FileNotFoundError:
            print(f"❌ Workout file '{filename}' not found")
            return None
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON in '{filename}'")
            return None
    
    def run_workout_program(self, workout_data):
        """Run a complete workout program from JSON with full tracking"""
        workout_name = workout_data.get('name', 'Custom Workout')
        
        # Start workout session
        self.start_workout_session(workout_name)
        
        print(f"\n🏋️‍♀️ Starting Workout: {workout_name}")
        if 'description' in workout_data:
            print(f"📋 {workout_data['description']}")
        
        total_exercises = len(workout_data['exercises'])
        
        for i, exercise_data in enumerate(workout_data['exercises'], 1):
            exercise_name = exercise_data['exercise']
            reps = exercise_data.get('reps', 10)
            sets = exercise_data.get('sets', 1)
            rest_time = exercise_data.get('rest', 30)
            
            if exercise_name not in self.exercises:
                print(f"⚠️ Skipping unknown exercise: {exercise_name}")
                continue
            
            print(f"\n📍 Exercise {i}/{total_exercises}: {self.exercises[exercise_name]['name']}")
            print(f"🎯 Target: {sets} sets x {reps} reps")
            
            for set_num in range(1, sets + 1):
                print(f"\n🏋️‍♀️ Set {set_num}/{sets}")
                input("Press Enter when ready...")
                
                self.run_workout(exercise_name, reps, sets, set_num)
                
                if set_num < sets:
                    print(f"😴 Rest for {rest_time} seconds...")
                    for countdown in range(rest_time, 0, -1):
                        print(f"   Rest: {countdown}s", end='\r')
                        time.sleep(1)
                    print("   Ready for next set!   ")
            
            print(f"✅ {self.exercises[exercise_name]['name']} complete!")
        
        print(f"\n🎉 Workout '{workout_name}' completed! Great job!")
        
        # Save workout report
        saved_file = self.save_workout_report()
        if saved_file:
            print(f"📊 Full workout report saved to: {saved_file}")
            
            # Ask if user wants to view the report
            view_report = input("\n📊 View detailed report? (y/n): ").strip().lower()
            if view_report == 'y':
                self.load_workout_report(saved_file)
    
    def show_exercises_by_category(self):
        """Show exercises organized by muscle group"""
        categories = {
            'Glutes': [],
            'Biceps/Upper Body': [],
            'Core/Abs': [],
            'Legs': [],
            'Pilates/Mobility': [],
            'Other': []
        }
        
        for key, exercise in self.exercises.items():
            muscle = exercise.get('primary_muscle', 'Other')
            if 'Glute' in muscle:
                categories['Glutes'].append((key, exercise))
            elif 'Bicep' in muscle or 'Chest' in muscle or 'Shoulder' in muscle:
                categories['Biceps/Upper Body'].append((key, exercise))
            elif 'Core' in muscle or 'Abs' in muscle or 'Oblique' in muscle:
                categories['Core/Abs'].append((key, exercise))
            elif 'Quad' in muscle or 'Hamstring' in muscle or 'Calves' in muscle:
                categories['Legs'].append((key, exercise))
            elif exercise.get('type') in ['Mobility', 'Stability']:
                categories['Pilates/Mobility'].append((key, exercise))
            else:
                categories['Other'].append((key, exercise))
        
        for category, exercises in categories.items():
            if exercises:
                print(f"\n🔥 {category.upper()}")
                print("-" * 40)
                for key, exercise in exercises:
                    equipment = exercise.get('equipment', 'Unknown')
                    difficulty = exercise.get('difficulty', 'Unknown')
                    print(f"  {key}: {exercise['name']} ({difficulty}, {equipment})")
    
    def create_sample_workout(self):
        """Create sample workout JSON file"""
        sample_workout = {
            "name": "Upper Body Strength",
            "description": "A beginner-friendly upper body workout focusing on biceps and shoulders",
            "exercises": [
                {
                    "exercise": "bicep_curls",
                    "sets": 3,
                    "reps": 12,
                    "rest": 45
                },
                {
                    "exercise": "shoulder_press",
                    "sets": 3,
                    "reps": 10,
                    "rest": 60
                },
                {
                    "exercise": "push_ups",
                    "sets": 2,
                    "reps": 8,
                    "rest": 30
                }
            ]
        }
        
        filename = "sample_workout.json"
        with open(filename, 'w') as f:
            json.dump(sample_workout, f, indent=2)
        
        print(f"✅ Sample workout saved as '{filename}'")
        return filename

def main():
    """Main function"""
    tracker = MoveNetWorkoutTracker()
    
    while True:
        print("\n" + "="*60)
        print("🏋️‍♀️ MOVENET WORKOUT TRACKER")
        print("="*60)
        print("Choose an option:")
        print("1. 💪 Single Exercise")
        print("2. 📋 Load Workout from JSON")
        print("3. 📚 Browse Exercises by Category")
        print("4. 📄 Create Sample Workout")
        print("5. 🔍 Search Exercise")
        print("6. 📹 Select Camera")
        print("7. 🎯 Test Camera View")
        print("8. 📊 View Workout Reports")
        print("9. ❌ Quit")
        print()
        
        choice = input("Enter your choice (1-9): ").strip()
        
        if choice == '1':
            # Single exercise mode
            print("\nAvailable exercises:")
            for key, exercise in list(tracker.exercises.items())[:10]:  # Show first 10
                print(f"  {key}: {exercise['name']}")
            print("  ... (type 'all' to see all exercises)")
            
            exercise_choice = input("\nChoose exercise: ").strip().lower()
            
            if exercise_choice == 'all':
                tracker.show_exercises_by_category()
                exercise_choice = input("\nChoose exercise: ").strip().lower()
            
            if exercise_choice in tracker.exercises:
                try:
                    reps = int(input("Target reps (default 10): ") or "10")
                    sets = int(input("Target sets (default 1): ") or "1")
                    
                    # Start a single exercise session
                    exercise_name = tracker.exercises[exercise_choice]['name']
                    tracker.start_workout_session(f"Single Exercise: {exercise_name}")
                    
                    # Run the exercise with all sets
                    for set_num in range(1, sets + 1):
                        if sets > 1:
                            print(f"\n🏋️‍♀️ Set {set_num}/{sets}")
                            input("Press Enter when ready...")
                        
                        tracker.run_workout(exercise_choice, reps, sets, set_num)
                        
                        if set_num < sets:
                            rest_time = 60  # Default rest time
                            print(f"😴 Rest for {rest_time} seconds...")
                            for countdown in range(rest_time, 0, -1):
                                print(f"   Rest: {countdown}s", end='\r')
                                time.sleep(1)
                            print("   Ready for next set!   ")
                    
                    # Save report
                    saved_file = tracker.save_workout_report()
                    if saved_file:
                        view_report = input("\n📊 View workout report? (y/n): ").strip().lower()
                        if view_report == 'y':
                            tracker.load_workout_report(saved_file)
                            
                except ValueError:
                    print("❌ Please enter a valid number")
            else:
                print("❌ Invalid exercise choice")
        
        elif choice == '2':
            # Load workout from JSON
            filename = input("Enter JSON workout filename: ").strip()
            if filename:
                workout_data = tracker.load_workout_from_json(filename)
                if workout_data:
                    tracker.run_workout_program(workout_data)
        
        elif choice == '3':
            # Browse exercises by category
            tracker.show_exercises_by_category()
        
        elif choice == '4':
            # Create sample workout
            sample_file = tracker.create_sample_workout()
            print(f"\nYou can now run this workout with option 2 using filename: {sample_file}")
        
        elif choice == '5':
            # Search exercise
            search_term = input("Search for exercise (name or muscle): ").strip().lower()
            found = []
            for key, exercise in tracker.exercises.items():
                if (search_term in exercise['name'].lower() or 
                    search_term in exercise.get('primary_muscle', '').lower()):
                    found.append((key, exercise))
            
            if found:
                print(f"\n🔍 Found {len(found)} exercises:")
                for key, exercise in found:
                    print(f"  {key}: {exercise['name']} ({exercise.get('primary_muscle', 'Unknown')})")
            else:
                print("❌ No exercises found")
        
        elif choice == '6':
            # Camera selection
            tracker.select_camera()
        
        elif choice == '7':
            # Camera view test
            tracker.test_camera_view()
        
        elif choice == '8':
            # View workout reports
            reports = tracker.list_workout_reports()
            if reports:
                try:
                    choice = input(f"\nEnter report number (1-{len(reports)}) or filename: ").strip()
                    if choice.isdigit() and 1 <= int(choice) <= len(reports):
                        selected_report = reports[int(choice) - 1]
                        tracker.load_workout_report(os.path.join("workout_reports", selected_report))
                    elif choice:
                        # Try as direct filename
                        if not choice.endswith('.json'):
                            choice += '.json'
                        tracker.load_workout_report(os.path.join("workout_reports", choice))
                except ValueError:
                    print("❌ Invalid selection")
        
        elif choice == '9':
            print("👋 Goodbye! Stay fit!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-9.")

if __name__ == "__main__":
    main()
