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
        self.primary_camera_index = 0  # Laptop camera (adjustable)
        self.secondary_camera_index = 1  # Overhead/sky camera (fixed)
        self.dual_camera_mode = False  # Enable dual camera tracking
        self.camera_setup = None  # Store current camera configuration

        # Exercise tracking - stricter parameters for better form validation
        self.rep_count = 0
        self.state = 'down'
        self.last_rep_time = 0
        self.min_rep_time = 1.2
        self.angle_history = []
        self.confidence_threshold = 0.4  # Increased from 0.3 - require higher confidence for strict form
        self.last_form_score = None  # Track last form score for display

        # HOLD REQUIREMENTS TRACKING - Critical for proper exercise execution
        self.position_hold_start = 0  # When we entered the target position
        self.current_hold_duration = 0  # How long we've been holding
        self.hold_requirements_met = False  # Did we meet the hold requirement
        self.target_position_stable = False  # Are we in the stable target position
        self.last_stable_angle = 0  # Last angle when position was stable
        self.position_stability_buffer = []  # Buffer to check position stability
        self.hold_angle_tolerance = 25  # Degrees of movement allowed during hold - more forgiving for vision jitter

        # HOLD PHASE MANAGEMENT - Separate from normal rep counting
        self.hold_phase_active = False  # Are we currently in a hold phase
        self.hold_phase_start_time = None  # When the hold phase started
        self.hold_phase_exercise = None  # Which exercise we're holding for
        self.rep_pending_hold = False  # Rep completed, waiting for hold

        # Progressive tracking system (enabled by default)
        self.progressive_states = []
        self.progression_thresholds = [180, 150, 90, 60]
        self.current_progression = 0
        self.progression_direction = 'down'
        self.strict_form_enabled = True
        self.progression_start_time = 0

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
        
        # Current exercise tracking
        self.current_exercise = None
        
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
                'hold_requirements': {
                    'top_hold': 3.0,  # Must hold at top for 3 seconds
                    'position_check': 'spine_straight',  # Must have straight spine
                    'angle_requirement': 90,  # Knees at 90 degrees
                    'stability_required': True  # Position must be stable during hold
                },
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
                'description': 'Back on bench, feet shoulder width, spine neutral',
                'hold_requirements': {
                    'top_hold': 3.0,  # Must hold at top for 3 seconds
                    'position_check': 'spine_straight',  # Must have straight spine
                    'angle_requirement': 90,  # Knees at 90 degrees
                    'stability_required': True  # Position must be stable during hold
                }
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
                'description': 'Full range, no swinging',
                'hold_requirements': {
                    'top_hold': 1.0,  # Reduced from 2.0 - Hold at peak contraction for 1 second only
                    'position_check': 'bicep_peak',  # Must be at peak bicep contraction
                    'angle_requirement': 50,  # Target angle to maintain
                    'stability_required': False,  # Disable strict stability requirement
                    'full_contraction': True  # Must reach full bicep contraction
                }
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
                'description': 'Elbows under shoulders',
                'hold_requirements': {
                    'position_hold': 10.0,  # Must hold plank for 10 seconds minimum
                    'position_check': 'plank_straight',  # Must maintain straight line
                    'angle_requirement': 180,  # Straight line from head to heels
                    'stability_required': True,  # Must be very stable
                    'core_engaged': True  # Core must be engaged
                }
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
                'description': 'Chest up, knees out',
                'hold_requirements': {
                    'bottom_hold': 3.0,  # Must hold at bottom for 3 seconds
                    'position_check': 'squat_depth',  # Must achieve proper depth
                    'angle_requirement': 90,  # Hip-knee-ankle at 90 degrees
                    'stability_required': True,  # Must be stable during hold
                    'knees_out': True  # Knees must track over toes
                }
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
                'description': 'Toes out, wide stance',
                'hold_requirements': {
                    'bottom_hold': 3.0,  # Must hold at bottom for 3 seconds
                    'position_check': 'squat_depth',  # Must achieve proper depth
                    'angle_requirement': 90,  # Hip-knee-ankle at 90 degrees
                    'stability_required': True,  # Must be stable during hold
                    'wide_stance': True  # Wide stance required
                }
            },
            'romanian_deadlift': {
                'name': 'Romanian Deadlift',
                'primary_muscle': 'Hamstrings/Glutes',
                'equipment': 'Dumbbell/Barbell',
                'difficulty': 'Intermediate',
                'type': 'Strength',
                'keypoints': ['left_hip', 'left_knee', 'left_ankle', 'left_shoulder'],
                'down_threshold': 80,    # Hip hinge down position
                'up_threshold': 160,     # Standing upright position
                'description': 'Soft knees, hinge at hips, keep back straight',
                'form_tips': [
                    '🦵 Keep knees slightly bent throughout',
                    '🍑 Push hips back, not down',
                    '📏 Lower until you feel hamstring stretch',
                    '🏋️ Keep weight close to legs',
                    '📐 Maintain neutral spine'
                ],
                'camera_setup': {
                    'primary': 'side',          # Side view essential for form
                    'secondary': 'front',       # Front view for symmetry
                    'primary_tracking': True,
                    'form_validation': 'both'
                }
            },
            'calf_raise': {
                'name': 'Calf Raise',
                'primary_muscle': 'Calves',
                'equipment': 'Bodyweight/Dumbbell',
                'difficulty': 'Beginner',
                'type': 'Strength',
                'keypoints': ['left_knee', 'left_ankle', 'left_ankle'],  # Will use special height calculation
                'down_threshold': 0.1,   # Flat foot position (low positive)
                'up_threshold': 1.2,     # Tippy toe position (high positive)
                'description': 'Hold at top (tippy toes)',
                'special_tracking': 'calf_height',  # Use ankle height instead of angle
                'hold_requirements': {
                    'top_hold': 5.0,  # Must hold on tippy toes for 5 seconds
                    'position_check': 'calf_extension',  # Must be on tippy toes
                    'height_requirement': 1.0,  # Target positive value for tippy toes
                    'stability_required': True,  # Must maintain balance
                    'full_extension': True  # Must reach full tippy toe position
                }
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
                'down_threshold': 170,  # STRICTER: More extended arm required
                'up_threshold': 50,     # STRICTER: More flexed arm required
                'description': 'Keep elbow close to body, full curl',
                'form_requirements': {
                    'elbow_stability': True,
                    'controlled_movement': True,
                    'full_range': True
                },
                'hold_requirements': {
                    'top_hold': 2.0,  # Must hold at top contraction for 2 seconds
                    'position_check': 'bicep_peak',  # Must achieve full contraction
                    'angle_requirement': 50,  # Full flexion angle
                    'stability_required': True,  # Must control the weight
                    'squeeze_muscle': True  # Must squeeze at the top
                },
                'min_rep_time': 1.5,  # Minimum time for one rep
                'camera_setup': {
                    'primary': 'side',
                    'secondary': 'front',
                    'form_validation': 'strict'
                }
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
                'down_threshold': 80,   # Arms at shoulder level (stricter)
                'up_threshold': 175,    # Arms fully extended overhead (stricter)
                'description': 'Press weights overhead, keep core tight',
                'form_requirements': {
                    'min_rep_time': 1.5,  # Minimum time for controlled movement
                    'requires_overhead': True,  # Must reach full overhead position
                    'validate_form': True,  # Enable form validation
                    'strict_range': True    # Require full range of motion
                }
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
            $synth.Speak("Starting workout");
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
    
    def enable_strict_form_validation(self, enabled=True):
        """Enable or disable strict progressive form validation"""
        self.strict_form_enabled = enabled
        if enabled:
            self.reset_progression()
        return enabled
    
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
        self.current_exercise = exercise_type  # Set current exercise for hold requirements
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

    def safe_angle_by_names(self, keypoints, names_triplet):
        """Compute angle for a (name1, name2, name3) triplet using available keypoints.
        Returns (angle, available) where available is True if all three had sufficient confidence.
        """
        try:
            idxs = [self.KEYPOINT_DICT[n] for n in names_triplet]
            pts = []
            for idx in idxs:
                y, x, conf = keypoints[idx]
                if conf < self.confidence_threshold:
                    return 0, False
                pts.append((y, x))
            return self.calculate_angle(pts[0], pts[1], pts[2]), True
        except Exception:
            return 0, False

    def spine_straight_score(self, keypoints, side='left'):
        """Estimate spine straightness using ear->shoulder->hip and shoulder->hip->knee.
        Returns score 0-100 based on how near 180° the torso chain angles are, or None if unreliable."""
        shoulder = 'left_shoulder' if side == 'left' else 'right_shoulder'
        hip = 'left_hip' if side == 'left' else 'right_hip'
        ear = 'left_ear' if side == 'left' else 'right_ear'
        knee = 'left_knee' if side == 'left' else 'right_knee'
        
        a1, ok1 = self.safe_angle_by_names(keypoints, (ear, shoulder, hip))
        a2, ok2 = self.safe_angle_by_names(keypoints, (shoulder, hip, knee))
        
        scores = []
        if ok1:
            # Much stricter scoring - penalize heavily for deviation from 180°
            deviation1 = abs(180 - a1)
            score1 = max(0, 100 - (deviation1 * 3))  # 3% penalty per degree
            scores.append(score1)
        if ok2:
            deviation2 = abs(180 - a2)
            score2 = max(0, 100 - (deviation2 * 3))  # 3% penalty per degree
            scores.append(score2)
            
        if not scores:
            return None  # No reliable data
        
        avg_score = sum(scores) / len(scores)
        return int(avg_score) if avg_score >= 20 else 10  # Minimum score for poor form

    def pelvis_tuck_score(self, keypoints, side='left'):
        """Rough proxy for posterior pelvic tilt: angle hip-shoulder-knee near 180 and hip below shoulder vertically.
        Returns 0-100 score, or None if unreliable. Conservative to avoid false positives."""
        shoulder = 'left_shoulder' if side == 'left' else 'right_shoulder'
        hip = 'left_hip' if side == 'left' else 'right_hip'
        knee = 'left_knee' if side == 'left' else 'right_knee'
        
        ang, ok = self.safe_angle_by_names(keypoints, (shoulder, hip, knee))
        if not ok:
            return None  # No reliable data
            
        # Much stricter pelvis evaluation
        deviation = abs(180 - ang)
        score = max(0, 100 - (deviation * 4))  # 4% penalty per degree
        return int(score) if score >= 15 else 5  # Minimum score for poor form
        # Favor straighter torso/leg alignment
        score = max(0, 100 - abs(180 - ang))
        return int(score)
    
    def knee_angle_score(self, keypoints, side='left', target_angle=90):
        """Check if knee is at proper angle (default 90 degrees for quadruped positions).
        Returns score 0-100, or None if unreliable data."""
        hip = 'left_hip' if side == 'left' else 'right_hip'
        knee = 'left_knee' if side == 'left' else 'right_knee'
        ankle = 'left_ankle' if side == 'left' else 'right_ankle'
        ang, ok = self.safe_angle_by_names(keypoints, (hip, knee, ankle))
        if not ok:
            return None  # No reliable data
        
        # Much stricter knee angle scoring
        deviation = abs(ang - target_angle)
        if deviation > 60:  # If deviation is extreme, very low score
            return 5
        
        score = max(0, 100 - (deviation * 3))  # 3% penalty per degree off (stricter)
        return int(score) if score >= 15 else 5  # Minimum score for poor form
    
    def shoulder_stability_score(self, keypoints):
        """Check shoulder alignment and stability for upper body exercises.
        Returns score 0-100, or None if unreliable data."""
        left_shoulder = keypoints[self.KEYPOINT_DICT['left_shoulder']]
        right_shoulder = keypoints[self.KEYPOINT_DICT['right_shoulder']]
        
        if left_shoulder[2] < self.confidence_threshold or right_shoulder[2] < self.confidence_threshold:
            return None  # Unreliable data
        
        # Much stricter shoulder level check
        shoulder_level_diff = abs(left_shoulder[0] - right_shoulder[0])
        if shoulder_level_diff > 0.05:  # If shoulders are very unlevel
            return 5
            
        level_score = max(0, 100 - (shoulder_level_diff * 2000))  # Very strict penalty
        return int(level_score) if level_score >= 20 else 10  # Minimum score for poor form
    
    def core_engagement_score(self, keypoints):
        """Estimate core engagement by checking torso stability.
        Returns score 0-100, or None if unreliable data."""
        try:
            # Use shoulder-hip-knee angle on both sides
            left_score = self.spine_straight_score(keypoints, 'left')
            right_score = self.spine_straight_score(keypoints, 'right')
            
            # Also check hip-shoulder alignment
            left_hip = keypoints[self.KEYPOINT_DICT['left_hip']]
            right_hip = keypoints[self.KEYPOINT_DICT['right_hip']]
            left_shoulder = keypoints[self.KEYPOINT_DICT['left_shoulder']]
            right_shoulder = keypoints[self.KEYPOINT_DICT['right_shoulder']]
            
            if (left_hip[2] > self.confidence_threshold and right_hip[2] > self.confidence_threshold and 
                left_shoulder[2] > self.confidence_threshold and right_shoulder[2] > self.confidence_threshold):
                
                # Much stricter hip level check
                hip_level = abs(left_hip[0] - right_hip[0])
                if hip_level > 0.05:  # If hips are very unlevel
                    hip_score = 5
                else:
                    hip_score = max(0, 100 - (hip_level * 2000))  # Very strict penalty
                
                # Only use valid scores (not None)
                valid_scores = [s for s in [left_score, right_score] if s is not None]
                if valid_scores:
                    valid_scores.append(hip_score)
                    avg_score = sum(valid_scores) / len(valid_scores)
                    return int(avg_score) if avg_score >= 15 else 5  # Minimum score for poor form
                else:
                    return None  # No reliable spine data
            
            # Fallback to spine scores only
            valid_scores = [s for s in [left_score, right_score] if s is not None]
            if not valid_scores:
                return None  # No reliable data
            
            avg_score = sum(valid_scores) / len(valid_scores)
            return int(avg_score) if avg_score >= 15 else 5  # Minimum score for poor form
            
        except Exception:
            return None  # Error indicates unreliable data
    
    def leg_alignment_score(self, keypoints, side='left'):
        """Check proper leg alignment (hip-knee-ankle in line).
        Returns score 0-100, or None if unreliable data."""
        hip = 'left_hip' if side == 'left' else 'right_hip'
        knee = 'left_knee' if side == 'left' else 'right_knee'
        ankle = 'left_ankle' if side == 'left' else 'right_ankle'
        
        ang, ok = self.safe_angle_by_names(keypoints, (hip, knee, ankle))
        if not ok:
            return None  # No reliable data
        
        # Much stricter leg alignment scoring
        deviation = abs(180 - ang)
        if deviation > 45:  # Extreme misalignment
            return 5
            
        score = max(0, 100 - (deviation * 3))  # 3% penalty per degree (stricter)
        return int(score) if score >= 20 else 10  # Minimum score for poor form
        return int(score)
    
    def squat_depth_score(self, keypoints, side='left'):
        """Check squat depth by measuring knee angle.
        Returns score 0-100, or None if unreliable data."""
        score = self.knee_angle_score(keypoints, side, target_angle=90)
        # For squats, deeper is generally better (closer to 90 degrees)
        return score
    
    def overhead_position_score(self, keypoints, side='left'):
        """Check if arms are properly overhead (for shoulder press, overhead squat).
        Returns score 0-100, or None if unreliable data."""
        shoulder = 'left_shoulder' if side == 'left' else 'right_shoulder'
        elbow = 'left_elbow' if side == 'left' else 'right_elbow'
        wrist = 'left_wrist' if side == 'left' else 'right_wrist'
        
        ang, ok = self.safe_angle_by_names(keypoints, (shoulder, elbow, wrist))
        if not ok:
            return None  # No reliable data
            
        # For overhead position, arm should be close to 180° (straight up)
        deviation = abs(180 - ang)
        if deviation > 45:  # Poor overhead position
            return 5
            
        score = max(0, 100 - (deviation * 3))  # 3% penalty per degree
        return int(score) if score >= 20 else 10  # Minimum score for poor form
    
    def squat_knee_tracking_score(self, keypoints, side='left'):
        """Check if knees track properly and don't go over toes in squats.
        Returns score 0-100, or None if unreliable data."""
        try:
            knee_key = 'left_knee' if side == 'left' else 'right_knee'
            ankle_key = 'left_ankle' if side == 'left' else 'right_ankle'
            
            knee_idx = self.KEYPOINT_DICT[knee_key]
            ankle_idx = self.KEYPOINT_DICT[ankle_key]
            
            if (keypoints[knee_idx][2] < self.confidence_threshold or 
                keypoints[ankle_idx][2] < self.confidence_threshold):
                return None  # Unreliable data
            
            # Get knee and ankle positions (y, x coordinates)
            knee_y, knee_x = keypoints[knee_idx][0], keypoints[knee_idx][1]
            ankle_y, ankle_x = keypoints[ankle_idx][0], keypoints[ankle_idx][1]
            
            # Check if knee goes over toe (knee_x should not exceed ankle_x significantly)
            horizontal_offset = abs(knee_x - ankle_x)
            
            # Penalize heavily if knee goes too far forward
            if horizontal_offset > 0.1:  # Knee significantly over toes
                return 5  # Very poor form
            elif horizontal_offset > 0.05:  # Moderate knee forward tracking
                return 30  # Poor form
            else:
                # Good knee tracking
                score = max(50, 100 - (horizontal_offset * 1000))
                return int(score)
                
        except Exception:
            return None  # Error indicates unreliable data
    
    def squat_depth_and_form_score(self, keypoints, side='left'):
        """Comprehensive squat form checking: depth + knee tracking + alignment.
        Returns score 0-100, or None if unreliable data."""
        try:
            # Get individual scores
            depth_score = self.squat_depth_score(keypoints, side)
            knee_tracking_score = self.squat_knee_tracking_score(keypoints, side)
            alignment_score = self.leg_alignment_score(keypoints, side)
            
            # Only use valid scores
            valid_scores = [s for s in [depth_score, knee_tracking_score, alignment_score] if s is not None]
            
            if len(valid_scores) < 2:  # Need at least 2 measurements
                return None
            
            # Weight the scores (knee tracking is most important for safety)
            if knee_tracking_score is not None and knee_tracking_score < 20:
                # Heavily penalize poor knee tracking (safety issue)
                return max(5, int(sum(valid_scores) / len(valid_scores) * 0.3))
            else:
                return int(sum(valid_scores) / len(valid_scores))
        except Exception:
            return None

    def deadlift_form_score(self, keypoints):
        """Comprehensive Romanian deadlift form validation: arms, hips, legs, spine.
        Returns score 0-100, or None if unreliable data."""
        # Check spine alignment (ear-shoulder-hip)
        spine_score = self.spine_straight_score(keypoints)
        
        # Check knee stability (should stay slightly bent, not lock out)
        left_knee_score = self.knee_angle_score(keypoints, 'left', target_angle=170)  # Slight bend
        right_knee_score = self.knee_angle_score(keypoints, 'right', target_angle=170)
        
        # Check hip hinge (shoulder-hip-knee should form proper hinge)
        hip_hinge_left, ok1 = self.safe_angle_by_names(keypoints, ('left_shoulder', 'left_hip', 'left_knee'))
        hip_hinge_right, ok2 = self.safe_angle_by_names(keypoints, ('right_shoulder', 'right_hip', 'right_knee'))
        
        hip_hinge_score = None
        if ok1 or ok2:
            # Use available side, prefer bilateral average
            if ok1 and ok2:
                avg_hinge = (hip_hinge_left + hip_hinge_right) / 2
            elif ok1:
                avg_hinge = hip_hinge_left
            else:
                avg_hinge = hip_hinge_right
                
            # Good deadlift hip hinge around 90-120 degrees
            if 90 <= avg_hinge <= 120:
                hip_hinge_score = 90
            elif 80 <= avg_hinge <= 130:
                hip_hinge_score = 70
            elif 70 <= avg_hinge <= 140:
                hip_hinge_score = 50
            else:
                hip_hinge_score = 20
        
        # Check arm position (should hang straight down)
        left_arm_score = None
        right_arm_score = None
        
        # Left arm: shoulder-elbow-wrist should be straight (~180°)
        left_arm_angle, ok_left_arm = self.safe_angle_by_names(keypoints, ('left_shoulder', 'left_elbow', 'left_wrist'))
        if ok_left_arm:
            deviation = abs(180 - left_arm_angle)
            left_arm_score = max(20, 100 - (deviation * 3))  # 3% penalty per degree
            
        # Right arm: shoulder-elbow-wrist should be straight (~180°)
        right_arm_angle, ok_right_arm = self.safe_angle_by_names(keypoints, ('right_shoulder', 'right_elbow', 'right_wrist'))
        if ok_right_arm:
            deviation = abs(180 - right_arm_angle)
            right_arm_score = max(20, 100 - (deviation * 3))  # 3% penalty per degree
        
        # Collect all valid scores
        valid_scores = []
        if spine_score is not None:
            valid_scores.append(spine_score)
        if left_knee_score is not None:
            valid_scores.append(left_knee_score)
        if right_knee_score is not None:
            valid_scores.append(right_knee_score)
        if hip_hinge_score is not None:
            valid_scores.append(hip_hinge_score)
        if left_arm_score is not None:
            valid_scores.append(left_arm_score)
        if right_arm_score is not None:
            valid_scores.append(right_arm_score)
            
        if len(valid_scores) < 3:  # Need at least 3 measurements for deadlift
            return None
            
        # Return average of all valid measurements
        return int(sum(valid_scores) / len(valid_scores))

    def frog_pump_pelvic_score(self, keypoints):
        """Check pelvic angle and knee separation for frog pumps.
        Returns score 0-100, or None if unreliable data."""
        # Check hip height (pelvis should lift up)
        left_hip = keypoints[self.KEYPOINT_DICT['left_hip']]
        right_hip = keypoints[self.KEYPOINT_DICT['right_hip']]
        left_shoulder = keypoints[self.KEYPOINT_DICT['left_shoulder']]
        right_shoulder = keypoints[self.KEYPOINT_DICT['right_shoulder']]
        
        if (left_hip[2] < self.confidence_threshold or right_hip[2] < self.confidence_threshold or
            left_shoulder[2] < self.confidence_threshold or right_shoulder[2] < self.confidence_threshold):
            return None
            
        # Calculate hip elevation relative to shoulders
        avg_hip_y = (left_hip[0] + right_hip[0]) / 2
        avg_shoulder_y = (left_shoulder[0] + right_shoulder[0]) / 2
        
        # In frog pump, hips should be elevated above shoulders when viewed from side
        hip_elevation = avg_shoulder_y - avg_hip_y  # Positive means hips are above shoulders
        
        if hip_elevation > 0.1:  # Good hip elevation
            elevation_score = 90
        elif hip_elevation > 0.05:  # Moderate elevation
            elevation_score = 70
        elif hip_elevation > 0:  # Slight elevation
            elevation_score = 50
        else:  # No elevation or hips below shoulders
            elevation_score = 20
            
        # Check knee separation (knees should be wide apart)
        left_knee = keypoints[self.KEYPOINT_DICT['left_knee']]
        right_knee = keypoints[self.KEYPOINT_DICT['right_knee']]
        
        if left_knee[2] < self.confidence_threshold or right_knee[2] < self.confidence_threshold:
            return elevation_score  # Return just elevation score if knees not visible
            
        # Calculate knee separation (wider is better for frog pumps)
        knee_separation = abs(left_knee[1] - right_knee[1])  # x-axis separation
        
        if knee_separation > 0.3:  # Wide knee separation
            separation_score = 90
        elif knee_separation > 0.2:  # Moderate separation
            separation_score = 70
        elif knee_separation > 0.1:  # Some separation
            separation_score = 50
        else:  # Knees too close together
            separation_score = 20
            
        # Combine elevation and separation scores
        return int((elevation_score + separation_score) / 2)

    def plank_stability_score(self, keypoints):
        """Check plank form: straight line from head to heels.
        Returns score 0-100, or None if unreliable data."""
        # Check full body alignment: shoulder-hip-ankle
        shoulder_hip_ankle_left, ok1 = self.safe_angle_by_names(keypoints, ('left_shoulder', 'left_hip', 'left_ankle'))
        shoulder_hip_ankle_right, ok2 = self.safe_angle_by_names(keypoints, ('right_shoulder', 'right_hip', 'right_ankle'))
        
        if not (ok1 or ok2):
            return None
            
        # Use available side or average
        if ok1 and ok2:
            avg_angle = (shoulder_hip_ankle_left + shoulder_hip_ankle_right) / 2
        elif ok1:
            avg_angle = shoulder_hip_ankle_left
        else:
            avg_angle = shoulder_hip_ankle_right
            
        # Good plank should be close to 180° (straight line)
        deviation = abs(180 - avg_angle)
        if deviation < 5:
            return 95  # Excellent plank form
        elif deviation < 10:
            return 85  # Good plank form
        elif deviation < 20:
            return 65  # Moderate plank form
        elif deviation < 30:
            return 40  # Poor plank form
        else:
            return 15  # Very poor plank form

    def lunge_stability_score(self, keypoints):
        """Check lunge form: front knee tracking, back leg position.
        Returns score 0-100, or None if unreliable data."""
        # For lunge, check front leg knee angle and back leg extension
        front_knee_score = self.knee_angle_score(keypoints, 'left', target_angle=90)
        back_leg_score = self.leg_alignment_score(keypoints)
        spine_score = self.spine_straight_score(keypoints)
        
        valid_scores = [s for s in [front_knee_score, back_leg_score, spine_score] if s is not None]
        
        if len(valid_scores) < 2:
            return None
            
        return int(sum(valid_scores) / len(valid_scores))

    def glute_bridge_form_score(self, keypoints):
        """Comprehensive glute bridge form validation: hip elevation, spine alignment, knee position.
        Returns score 0-100, or None if unreliable data."""
        spine_score = self.spine_straight_score(keypoints)
        pelvis_score = self.pelvis_tuck_score(keypoints)
        knee_score = self.knee_angle_score(keypoints, 'left', target_angle=90)
        alignment_score = self.leg_alignment_score(keypoints)
        
        valid_scores = [s for s in [spine_score, pelvis_score, knee_score, alignment_score] if s is not None]
        
        if len(valid_scores) < 2:
            return None
            
        return int(sum(valid_scores) / len(valid_scores))

    def get_exercise_definition(self, exercise_name):
        """Get the full exercise definition from exercises list"""
        return self.exercises.get(exercise_name)
    
    def start_hold_phase(self, exercise_type, current_time):
        """Start a hold phase after rep completion"""
        self.hold_phase_active = True
        self.hold_phase_start_time = current_time
        self.hold_phase_exercise = exercise_type
        self.rep_pending_hold = False
        # Reset hold tracking for this phase
        self.position_hold_start = 0
        self.current_hold_duration = 0
        self.hold_requirements_met = False
        self.target_position_stable = False
        self.position_stability_buffer = []
        
    def process_hold_phase(self, current_angle, keypoints, current_time):
        """Process the hold phase separately from rep counting"""
        if not self.hold_phase_active:
            return False, 0, False
            
        # Check hold requirements for the current exercise
        hold_met, hold_duration, position_stable = self.check_hold_requirements(
            self.hold_phase_exercise, current_angle, keypoints, current_time)
        
        if hold_met:
            # Hold completed! Exit hold phase
            self.hold_phase_active = False
            self.hold_phase_start_time = None
            self.hold_phase_exercise = None
            self.reset_hold_tracking()
            self.speak(f"Hold completed! Held for {hold_duration:.1f} seconds")
            return True, hold_duration, position_stable
        
        return False, hold_duration, position_stable
    
    def check_hold_requirements(self, exercise_type, current_angle, keypoints, current_time):
        """Check if hold requirements are met for the current exercise.
        Returns (hold_met, hold_duration, position_stable)"""
        
        exercise = self.exercises.get(exercise_type, {})
        hold_reqs = exercise.get('hold_requirements', {})
        
        # DEFAULT HOLD REQUIREMENTS for exercises without specific ones
        if not hold_reqs:
            # Apply universal hold requirements based on exercise type
            if 'squat' in exercise_type.lower():
                hold_reqs = {'bottom_hold': 3.0, 'position_check': 'squat_depth', 'angle_requirement': 90, 'stability_required': True}
            elif 'bridge' in exercise_type.lower() or 'thrust' in exercise_type.lower():
                hold_reqs = {'top_hold': 3.0, 'position_check': 'spine_straight', 'angle_requirement': 90, 'stability_required': True}
            elif 'calf' in exercise_type.lower():
                hold_reqs = {'top_hold': 5.0, 'position_check': 'calf_extension', 'height_requirement': 1.0, 'stability_required': True}
            elif 'plank' in exercise_type.lower():
                hold_reqs = {'position_hold': 10.0, 'position_check': 'plank_straight', 'angle_requirement': 180, 'stability_required': True}
            elif 'curl' in exercise_type.lower():
                hold_reqs = {'top_hold': 1.0, 'position_check': 'bicep_peak', 'angle_requirement': 50, 'stability_required': False}
            elif 'press' in exercise_type.lower():
                hold_reqs = {'top_hold': 2.0, 'position_check': 'overhead_extension', 'angle_requirement': 175, 'stability_required': True}
            elif 'deadlift' in exercise_type.lower():
                hold_reqs = {'bottom_hold': 2.0, 'position_check': 'hip_hinge', 'angle_requirement': 110, 'stability_required': True}
            elif 'lunge' in exercise_type.lower():
                hold_reqs = {'bottom_hold': 2.0, 'position_check': 'lunge_depth', 'angle_requirement': 90, 'stability_required': True}
            elif 'push' in exercise_type.lower():
                hold_reqs = {'bottom_hold': 1.5, 'position_check': 'push_depth', 'angle_requirement': 90, 'stability_required': True}
            else:
                # Default for any other exercise
                hold_reqs = {'hold_time': 2.0, 'position_check': 'angle', 'angle_requirement': 90, 'stability_required': True}
        
        # Get hold requirements
        required_hold_time = hold_reqs.get('top_hold', hold_reqs.get('bottom_hold', hold_reqs.get('position_hold', hold_reqs.get('hold_time', 2.0))))
        position_check = hold_reqs.get('position_check', 'angle')
        angle_requirement = hold_reqs.get('angle_requirement', 90)
        stability_required = hold_reqs.get('stability_required', True)
        
        # Check if we're in the target position
        in_target_position = False
        
        if position_check == 'spine_straight':
            # For glute bridges - check spine alignment and hip elevation
            spine_score = self.spine_straight_score(keypoints)
            if spine_score and spine_score >= 70:  # Good spine alignment
                # Check if hips are elevated (angle near target)
                angle_diff = abs(current_angle - angle_requirement)
                in_target_position = angle_diff <= self.hold_angle_tolerance
        
        elif position_check == 'squat_depth':
            # For squats - check if at proper depth
            knee_score = self.knee_angle_score(keypoints, 'left', target_angle=angle_requirement)
            if knee_score and knee_score >= 60:  # Good knee position
                # Check squat depth
                angle_diff = abs(current_angle - angle_requirement)
                in_target_position = angle_diff <= self.hold_angle_tolerance
        
        elif position_check == 'calf_extension':
            # For calf raises - check ankle height for tippy toe position
            # Determine which side to track
            if hasattr(self, 'current_tracking_side') and self.current_tracking_side:
                side = self.current_tracking_side
            else:
                side = 'left'  # Default to left
            
            calf_metric = self.calculate_calf_raise_metric(keypoints, side)
            if calf_metric is not None:
                # For tippy toes, metric should be above threshold (positive value)
                height_good = calf_metric >= 1.0  # Good tippy toe elevation
                
                # Additional stability check - make sure height is stable
                if hasattr(self, 'prev_calf_metric'):
                    height_stability = abs(calf_metric - self.prev_calf_metric) < 0.3  # Stability tolerance
                else:
                    height_stability = True
                
                self.prev_calf_metric = calf_metric
                in_target_position = height_good and height_stability
            else:
                in_target_position = False
                
        elif position_check == 'plank_straight':
            # For planks - check straight line alignment
            plank_score = self.plank_stability_score(keypoints)
            if plank_score and plank_score >= 70:  # Good plank alignment
                angle_diff = abs(current_angle - angle_requirement)
                in_target_position = angle_diff <= self.hold_angle_tolerance
            else:
                in_target_position = False
                
        elif position_check == 'bicep_peak':
            # For bicep curls - check peak contraction (reasonably forgiving)
            shoulder_score = self.shoulder_stability_score(keypoints)
            
            # Make requirements reasonably forgiving
            if shoulder_score is None or shoulder_score >= 40:  # Reasonable shoulder requirement
                # Check if at peak contraction angle
                angle_diff = abs(current_angle - angle_requirement)
                
                # Use reasonable tolerance for bicep curls
                bicep_tolerance = 30  # Reasonable tolerance
                in_target_position = angle_diff <= bicep_tolerance
            else:
                in_target_position = False
                
        elif position_check == 'overhead_extension':
            # For overhead press - check full extension
            overhead_score = self.overhead_position_score(keypoints)
            if overhead_score and overhead_score >= 70:  # Good overhead position
                angle_diff = abs(current_angle - angle_requirement)
                in_target_position = angle_diff <= self.hold_angle_tolerance
            else:
                in_target_position = False
                
        elif position_check == 'hip_hinge':
            # For deadlifts - check proper hip hinge
            spine_score = self.spine_straight_score(keypoints)
            knee_score = self.knee_angle_score(keypoints, 'left', target_angle=170)  # Slight knee bend
            if spine_score and spine_score >= 60 and knee_score and knee_score >= 60:
                angle_diff = abs(current_angle - angle_requirement)
                in_target_position = angle_diff <= self.hold_angle_tolerance
            else:
                in_target_position = False
                
        elif position_check == 'lunge_depth':
            # For lunges - check proper depth
            knee_score = self.knee_angle_score(keypoints, 'left', target_angle=90)
            if knee_score and knee_score >= 60:  # Good knee position
                angle_diff = abs(current_angle - angle_requirement)
                in_target_position = angle_diff <= self.hold_angle_tolerance
            else:
                in_target_position = False
                
        elif position_check == 'push_depth':
            # For push-ups - check bottom position depth
            plank_score = self.plank_stability_score(keypoints)
            if plank_score and plank_score >= 60:  # Good plank form
                angle_diff = abs(current_angle - angle_requirement)
                in_target_position = angle_diff <= self.hold_angle_tolerance
            else:
                in_target_position = False
        
        else:
            # Default angle check
            angle_diff = abs(current_angle - angle_requirement)
            in_target_position = angle_diff <= self.hold_angle_tolerance
        
        # Track position stability if required
        if stability_required:
            self.position_stability_buffer.append(current_angle)
            if len(self.position_stability_buffer) > 10:  # Keep last 10 readings
                self.position_stability_buffer.pop(0)
            
            # Check if position is stable (low variance)
            if len(self.position_stability_buffer) >= 5:
                variance = np.var(self.position_stability_buffer)
                position_stable = variance < 300  # More forgiving variance threshold for vision jitter
            else:
                position_stable = False
        else:
            position_stable = True
        
        # Update hold tracking
        if in_target_position and position_stable:
            if not self.target_position_stable:
                # Just entered stable target position
                self.position_hold_start = current_time
                self.target_position_stable = True
                self.current_hold_duration = 0
            else:
                # Continue holding
                self.current_hold_duration = current_time - self.position_hold_start
        else:
            # Not in target position or not stable - reset hold
            self.target_position_stable = False
            self.current_hold_duration = 0
            self.position_hold_start = 0
        
        # Check if hold requirement is met
        hold_met = self.current_hold_duration >= required_hold_time
        
        return hold_met, self.current_hold_duration, position_stable

    def reset_hold_tracking(self):
        """Reset hold tracking variables when starting new rep"""
        self.position_hold_start = 0
        self.current_hold_duration = 0
        self.hold_requirements_met = False
        self.target_position_stable = False
        self.position_stability_buffer = []
                

       
    
    def validate_progressive_movement(self, current_angle, exercise_type):
        """
        Progressive movement validation for strict form checking
        Ensures movement follows proper sequence based on exercise type
        """
        if not self.strict_form_enabled:
            return True, "Progressive validation disabled"
        
        current_time = time.time()
        
        # Exercise-specific thresholds and patterns
        if exercise_type == 'bicep_curls':
            self.progression_thresholds = [170, 140, 90, 50]  # Extended to contracted
            self.progression_direction_pattern = 'contract_extend'  # Contract then extend
        elif exercise_type == 'shoulder_press':
            self.progression_thresholds = [80, 100, 140, 175]  # Shoulder level to overhead
            self.progression_direction_pattern = 'extend_contract'  # Extend then contract
        elif exercise_type in ['squat', 'goblet_squat', 'sumo_squat']:
            self.progression_thresholds = [170, 140, 110, 90]  # Standing to deep squat
            self.progression_direction_pattern = 'descend_ascend'  # Down then up
        elif exercise_type in ['glute_bridge', 'hip_thrust']:
            self.progression_thresholds = [140, 155, 165, 175]  # Lying to bridge
            self.progression_direction_pattern = 'lift_lower'  # Lift then lower
        elif exercise_type in ['fire_hydrant', 'quadruped_hip_abduction']:
            self.progression_thresholds = [160, 140, 120, 100]  # Leg close to abducted
            self.progression_direction_pattern = 'abduct_adduct'  # Out then in
        elif exercise_type in ['romanian_deadlift', 'rdl']:
            self.progression_thresholds = [170, 150, 130, 110]  # Standing to hip hinge
            self.progression_direction_pattern = 'hinge_return'  # Hinge then return
        elif exercise_type in ['push_up', 'pushup']:
            self.progression_thresholds = [170, 140, 110, 90]  # Extended to lowered
            self.progression_direction_pattern = 'lower_push'  # Lower then push
        elif exercise_type in ['lunge', 'reverse_lunge', 'forward_lunge']:
            self.progression_thresholds = [170, 140, 110, 90]  # Standing to lunge
            self.progression_direction_pattern = 'descend_ascend'  # Down then up
        else:
            self.progression_thresholds = [180, 150, 90, 60]  # Default progression
            self.progression_direction_pattern = 'contract_extend'  # Default pattern
        
        # Initialize progression tracking
        if self.current_progression == 0 and self.progression_direction == 'down':
            self.progression_start_time = current_time
            self.progressive_states = []
        
        # Track progression based on direction and exercise pattern
        if self.progression_direction == 'down':
            # Moving through the first half of the movement
            target_angle = self.progression_thresholds[self.current_progression]
            
            # Determine if we're moving in the right direction based on exercise pattern
            if self.progression_direction_pattern in ['contract_extend', 'descend_ascend', 'hinge_return', 'lower_push']:
                movement_condition = current_angle <= target_angle
            else:  # extend_contract, lift_lower, abduct_adduct
                movement_condition = current_angle >= target_angle
            
            if movement_condition:
                self.progressive_states.append({
                    'stage': self.current_progression,
                    'angle': current_angle,
                    'time': current_time,
                    'threshold': target_angle
                })
                self.current_progression += 1
                
                # Check if we've completed the down phase
                if self.current_progression >= len(self.progression_thresholds):
                    self.progression_direction = 'up'
                    self.current_progression = len(self.progression_thresholds) - 2  # Start going back up
                    
        elif self.progression_direction == 'up':
            # Moving through the second half of the movement (return phase)
            target_angle = self.progression_thresholds[self.current_progression]
            
            # Determine return movement condition
            if self.progression_direction_pattern in ['contract_extend', 'descend_ascend', 'hinge_return', 'lower_push']:
                movement_condition = current_angle >= target_angle
            else:  # extend_contract, lift_lower, abduct_adduct
                movement_condition = current_angle <= target_angle
            
            if movement_condition:
                self.progressive_states.append({
                    'stage': self.current_progression,
                    'angle': current_angle,
                    'time': current_time,
                    'threshold': target_angle
                })
                self.current_progression -= 1
                
                # Check if we've completed the full rep
                if self.current_progression < 0:
                    # Full rep completed with proper progression
                    total_time = current_time - self.progression_start_time
                    self.reset_progression()
                    return True, f"Valid rep completed in {total_time:.1f}s"
        
        # Check for progression timeout (too slow)
        if current_time - self.progression_start_time > 15.0:  # 15 second timeout for complex exercises
            self.reset_progression()
            return False, "Movement too slow - progression timeout"
        
        # Check for skipped stages (too fast/jerky movement)
        if len(self.progressive_states) >= 2:
            last_two = self.progressive_states[-2:]
            stage_gap = abs(last_two[1]['stage'] - last_two[0]['stage'])
            time_gap = last_two[1]['time'] - last_two[0]['time']
            
            if stage_gap > 1 and time_gap < 0.4:  # Skipped stage too quickly
                self.reset_progression()
                return False, "Movement too jerky - skipped progression stage"
        
        return None, f"Progressing: Stage {self.current_progression}, Direction: {self.progression_direction}"
    
    def reset_progression(self):
        """Reset progressive tracking state"""
        self.current_progression = 0
        self.progression_direction = 'down'
        self.progressive_states = []
        self.progression_start_time = 0

    def calculate_calf_raise_metric(self, keypoints, side='left'):
        """Calculate calf raise metric using ankle height relative to knee"""
        try:
            if side == 'left':
                knee = keypoints[self.KEYPOINT_DICT['left_knee']]
                ankle = keypoints[self.KEYPOINT_DICT['left_ankle']]
            else:
                knee = keypoints[self.KEYPOINT_DICT['right_knee']]
                ankle = keypoints[self.KEYPOINT_DICT['right_ankle']]
            
            # Check confidence
            if (knee[2] < self.confidence_threshold or 
                ankle[2] < self.confidence_threshold):
                return None
            
            # Calculate vertical distance between knee and ankle
            # In image coordinates: smaller y = higher position
            # When on tippy toes, ankle y should be SMALLER (higher up)
            vertical_distance = knee[1] - ankle[1]  # Positive when ankle is above knee
            
            # Scale up for proper direction (positive values for tippy toes)
            calf_metric = vertical_distance * 100  # Positive values, tippy toes = positive
            
            # Add smoothing to reduce fluctuations
            if not hasattr(self, 'calf_metric_history'):
                self.calf_metric_history = []
            
            self.calf_metric_history.append(calf_metric)
            if len(self.calf_metric_history) > 5:  # Keep last 5 readings
                self.calf_metric_history.pop(0)
            
            # Return smoothed average
            smoothed_metric = sum(self.calf_metric_history) / len(self.calf_metric_history)
            return smoothed_metric
            
        except (IndexError, KeyError, ZeroDivisionError):
            return None
    
    def track_exercise(self, exercise_type, keypoints):
        """Simple, reliable exercise tracking with bilateral support"""
        if exercise_type not in self.exercises:
            return False, 0, 0
        
        # Set current exercise for tracking
        self.current_exercise = exercise_type
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
                
            elif exercise_type == 'calf_raise':
                # For calf raise: use ankle height relative to knee
                # Determine which side to track
                if hasattr(self, 'current_tracking_side') and self.current_tracking_side:
                    side = self.current_tracking_side
                else:
                    side = 'left'  # Default to left
                
                calf_metric = self.calculate_calf_raise_metric(keypoints, side)
                if calf_metric is None:
                    return False, 0, 0
                
                angle = calf_metric  # Use the height metric as our "angle"
                
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
            
            # Global progressive validation gate (default ON). We'll compute a primary angle for progression
            progression_result = None
            if self.strict_form_enabled:
                progression_result, _ = self.validate_progressive_movement(smoothed_angle, exercise_type)

            # Different logic for different exercises - COMPREHENSIVE VALIDATION FOR ALL
            if exercise_type in ['bicep_curls', 'bicep_curl', 'hammer_curl', 'concentration_curl', 'zottman_curl', 'supinated_curl_to_press']:
                # Enhanced bicep curls with comprehensive form validation (ALL curl variations)
                core_score = self.core_engagement_score(keypoints)
                shoulder_score = self.shoulder_stability_score(keypoints)
                spine_score = self.spine_straight_score(keypoints)
                
                # Only use valid (non-None) scores and be extremely strict
                valid_scores = [s for s in [core_score, shoulder_score, spine_score] if s is not None]
                
                if len(valid_scores) < 2:  # Need at least 2 reliable measurements
                    combined_form_score = 5  # Very poor form if can't track properly
                else:
                    combined_form_score = int(sum(valid_scores) / len(valid_scores))
                    # Additional penalty for wild movements - if too few valid scores, lower the result
                    if len(valid_scores) < 3:
                        combined_form_score = int(combined_form_score * 0.5)  # 50% penalty for missing data
                    
                    # Check for movement stability by looking at angle history variance
                    if len(self.angle_history) >= 3:
                        angle_variance = np.var(self.angle_history)
                        if angle_variance > 500:  # High variance indicates erratic movement
                            combined_form_score = int(combined_form_score * 0.3)  # 70% penalty for erratic movement
                        elif angle_variance > 200:  # Moderate variance
                            combined_form_score = int(combined_form_score * 0.6)  # 40% penalty
                
                # Ensure minimum score constraints for poor form
                if combined_form_score < 10:
                    combined_form_score = max(5, combined_form_score)
                
                if progression_result is True:
                    # Full progressive rep completed - complete rep immediately (NO HOLD)
                    rep_completed = True
                    self.last_rep_time = current_time
                    form_score = combined_form_score
                    self.state = 'down'  # Reset to down state
                    # Reset progression for next rep
                    self.current_progression = 0
                    self.progression_direction = 'down'
                    self.progressive_states = []
                        
                elif progression_result is False:
                    # Progressive validation failed - reset state
                    self.state = 'down'
                    form_score = 10  # Very low score for failed progression
                    self.reset_hold_tracking()  # Reset hold tracking
                    
                else:
                    # Still progressing - use fallback logic for basic state tracking
                    if self.state == 'down' and smoothed_angle < exercise['up_threshold'] - 5:  # Stricter threshold
                        if current_time - self.last_rep_time > 0.8:  # Longer minimum time
                            self.state = 'up'
                    elif self.state == 'up' and smoothed_angle > exercise['down_threshold'] + 10:  # Stricter threshold
                        if current_time - self.last_rep_time > 1.5 and combined_form_score >= 50:
                            # Complete rep immediately (NO HOLD)
                            rep_completed = True
                            self.last_rep_time = current_time
                            form_score = combined_form_score
                            self.state = 'down'
                            # Only count rep if not using progressive tracking or as backup AND hold requirement met
                            if not self.strict_form_enabled:
                                rep_completed = True
                                self.last_rep_time = current_time
                                form_score = combined_form_score
                                self.reset_hold_tracking()  # Reset for next rep
                            self.state = 'down'
                        
            elif exercise_type == 'shoulder_press':
                # Enhanced shoulder press with comprehensive form validation
                shoulder_score = self.shoulder_stability_score(keypoints)
                overhead_score = self.overhead_position_score(keypoints)
                core_score = self.core_engagement_score(keypoints)
                spine_score = self.spine_straight_score(keypoints)
                
                # Only use valid (non-None) scores and be extremely strict
                valid_scores = [s for s in [shoulder_score, overhead_score, core_score, spine_score] if s is not None]
                
                if len(valid_scores) < 2:  # Need at least 2 reliable measurements
                    combined_form_score = 5  # Very poor form if can't track properly
                else:
                    combined_form_score = int(sum(valid_scores) / len(valid_scores))
                    # Additional penalty for wild movements
                    if len(valid_scores) < 3:
                        combined_form_score = int(combined_form_score * 0.5)  # 50% penalty for missing data
                    
                    # Check for movement stability
                    if len(self.angle_history) >= 3:
                        angle_variance = np.var(self.angle_history)
                        if angle_variance > 500:  # High variance indicates erratic movement
                            combined_form_score = int(combined_form_score * 0.3)  # 70% penalty
                        elif angle_variance > 200:  # Moderate variance
                            combined_form_score = int(combined_form_score * 0.6)  # 40% penalty
                
                # Ensure minimum score constraints
                if combined_form_score < 10:
                    combined_form_score = max(5, combined_form_score)
                
                # STRICT: Shoulder press requires full overhead extension and controlled movement
                if self.state == 'down' and smoothed_angle > exercise['up_threshold'] - 5:  # Must reach almost full extension
                    if current_time - self.last_rep_time > 1.0:  # Longer pause for overhead position
                        self.state = 'up'
                elif self.state == 'up' and smoothed_angle < exercise['down_threshold'] + 10:  # Return to shoulder level
                    if current_time - self.last_rep_time > 1.5 and combined_form_score >= 40:  # Stricter form requirement
                        rep_completed = True
                        self.last_rep_time = current_time
                        form_score = combined_form_score
                        self.state = 'down'
                        
            elif exercise_type == 'romanian_deadlift':
                # Enhanced Romanian deadlift with FULL-BODY comprehensive form validation (arms, hips, legs, spine)
                deadlift_score = self.deadlift_form_score(keypoints)  # New comprehensive method
                knee_score = self.knee_angle_score(keypoints, None, target_angle=170)  # Slight knee bend
                alignment_score = self.leg_alignment_score(keypoints)
                spine_score = self.spine_straight_score(keypoints)
                core_score = self.core_engagement_score(keypoints)
                
                # Only use valid (non-None) scores and be extremely strict
                valid_scores = [s for s in [deadlift_score, knee_score, alignment_score, spine_score, core_score] if s is not None]
                
                if len(valid_scores) < 3:  # Need at least 3 reliable measurements for deadlift
                    combined_form_score = 5  # Very poor form if can't track properly
                else:
                    combined_form_score = int(sum(valid_scores) / len(valid_scores))
                    # Additional penalty for wild movements
                    if len(valid_scores) < 4:
                        combined_form_score = int(combined_form_score * 0.5)  # 50% penalty for missing data
                    
                    # Check for movement stability
                    if len(self.angle_history) >= 3:
                        angle_variance = np.var(self.angle_history)
                        if angle_variance > 500:  # High variance indicates erratic movement
                            combined_form_score = int(combined_form_score * 0.3)  # 70% penalty
                        elif angle_variance > 200:  # Moderate variance
                            combined_form_score = int(combined_form_score * 0.6)  # 40% penalty
                
                # Ensure minimum score constraints
                if combined_form_score < 10:
                    combined_form_score = max(5, combined_form_score)
                
                # Romanian deadlift: up = standing (large angle), down = hip hinge (small angle)
                # Focus on controlled movement and proper range
                if self.state == 'up' and smoothed_angle < exercise['down_threshold'] + 20:
                    self.state = 'down'
                elif self.state == 'down' and smoothed_angle > exercise['up_threshold'] - 20:
                    if current_time - self.last_rep_time > 2.0 and combined_form_score >= 50:  # Slower movement, stricter form
                        rep_completed = True
                        self.last_rep_time = current_time
                        form_score = combined_form_score
                    self.state = 'up'
                    
            elif exercise_type == 'frog_pump':
                # Enhanced frog pump with PELVIC ANGLE and knee separation tracking
                frog_pump_score = self.frog_pump_pelvic_score(keypoints)  # New pelvic angle method
                spine_score = self.spine_straight_score(keypoints)
                pelvis_score = self.pelvis_tuck_score(keypoints)
                core_score = self.core_engagement_score(keypoints)
                
                # Only use valid (non-None) scores and be extremely strict
                valid_scores = [s for s in [frog_pump_score, spine_score, pelvis_score, core_score] if s is not None]
                
                if len(valid_scores) < 2:  # Need at least 2 reliable measurements
                    combined_form_score = 5  # Very poor form if can't track properly
                else:
                    combined_form_score = int(sum(valid_scores) / len(valid_scores))
                    # Additional penalty for wild movements
                    if len(valid_scores) < 3:
                        combined_form_score = int(combined_form_score * 0.5)  # 50% penalty for missing data
                    
                    # Check for movement stability
                    if len(self.angle_history) >= 3:
                        angle_variance = np.var(self.angle_history)
                        if angle_variance > 500:  # High variance indicates erratic movement
                            combined_form_score = int(combined_form_score * 0.3)  # 70% penalty
                        elif angle_variance > 200:  # Moderate variance
                            combined_form_score = int(combined_form_score * 0.6)  # 40% penalty
                
                # Ensure minimum score constraints
                if combined_form_score < 10:
                    combined_form_score = max(5, combined_form_score)

                if progression_result is True:
                    rep_completed = True
                    self.last_rep_time = current_time
                    form_score = combined_form_score
                    self.state = 'down'
                elif progression_result is False:
                    self.state = 'down'
                    form_score = 10
                else:
                    # Frog pump: down = hips down, up = hips up with knees wide
                    if self.state == 'down' and smoothed_angle > 130:  # Hips up, knees wide
                        if current_time - self.last_rep_time > 0.8:
                            self.state = 'up'
                    elif self.state == 'up' and smoothed_angle < 90:  # Hips down, knees closer
                        if current_time - self.last_rep_time > 1.5 and combined_form_score >= 40:
                            rep_completed = True
                            self.last_rep_time = current_time
                            form_score = combined_form_score
                            self.state = 'down'
                            
            elif exercise_type in ['glute_bridge', 'hip_thrust', 'single_leg_glute_bridge']:
                # Enhanced glute bridges with comprehensive form validation
                glute_bridge_score = self.glute_bridge_form_score(keypoints)  # New comprehensive method
                spine_score = self.spine_straight_score(keypoints)
                pelvis_score = self.pelvis_tuck_score(keypoints)
                knee_score = self.knee_angle_score(keypoints, None, target_angle=90)
                alignment_score = self.leg_alignment_score(keypoints)
                
                # Only use valid (non-None) scores and be extremely strict
                valid_scores = [s for s in [glute_bridge_score, spine_score, pelvis_score, knee_score, alignment_score] if s is not None]
                
                if len(valid_scores) < 2:  # Need at least 2 reliable measurements
                    combined_form_score = 5  # Very poor form if can't track properly
                else:
                    combined_form_score = int(sum(valid_scores) / len(valid_scores))
                    # Additional penalty for wild movements
                    if len(valid_scores) < 3:
                        combined_form_score = int(combined_form_score * 0.5)  # 50% penalty for missing data
                    
                    # Check for movement stability
                    if len(self.angle_history) >= 3:
                        angle_variance = np.var(self.angle_history)
                        if angle_variance > 500:  # High variance indicates erratic movement
                            combined_form_score = int(combined_form_score * 0.3)  # 70% penalty
                        elif angle_variance > 200:  # Moderate variance
                            combined_form_score = int(combined_form_score * 0.6)  # 40% penalty
                
                # Ensure minimum score constraints
                if combined_form_score < 10:
                    combined_form_score = max(5, combined_form_score)

                if progression_result is True:
                    # Check hold requirements for glute bridge
                    hold_met, hold_duration, position_stable = self.check_hold_requirements(
                        self.current_exercise, smoothed_angle, keypoints, current_time)
                    
                    if hold_met:
                        rep_completed = True
                        self.last_rep_time = current_time
                        form_score = combined_form_score
                        self.state = 'down'  # Reset to down state
                        self.reset_hold_tracking()
                        # Force state transition by resetting progression
                        self.current_progression = 0
                        self.progression_direction = 'down'
                        self.progressive_states = []
                        self.speak(f"Glute bridge held for {hold_duration:.1f} seconds! Rep completed")
                    else:
                        # Still need to hold position longer
                        form_score = max(25, combined_form_score)  # Penalize for incomplete hold
                        
                elif progression_result is False:
                    self.state = 'down'
                    form_score = 10
                    self.reset_hold_tracking()
                else:
                    # Check hold requirements during movement
                    hold_met, hold_duration, position_stable = self.check_hold_requirements(
                        self.current_exercise, smoothed_angle, keypoints, current_time)
                    
                    # Basic angle thresholds with form validation AND HOLD REQUIREMENTS
                    if self.state == 'down' and smoothed_angle > 160:
                        if current_time - self.last_rep_time > 0.8:
                            self.state = 'up'
                    elif self.state == 'up' and smoothed_angle < 120:
                        if current_time - self.last_rep_time > 1.5 and combined_form_score >= 40 and hold_met:  # HOLD REQUIREMENT ADDED
                            rep_completed = True
                            self.last_rep_time = current_time
                            form_score = combined_form_score
                            self.state = 'down'
                            self.reset_hold_tracking()

            elif exercise_type in ['squat', 'goblet_squat', 'sumo_squat', 'bulgarian_split_squat']:
                # Enhanced squat with comprehensive whole-body form validation
                squat_form_score = self.squat_depth_and_form_score(keypoints)  # Includes knee tracking
                knee_score = self.knee_angle_score(keypoints, None, target_angle=90)
                spine_score = self.spine_straight_score(keypoints)
                core_score = self.core_engagement_score(keypoints)
                
                # Check both sides for knee tracking (safety critical)
                left_knee_tracking = self.squat_knee_tracking_score(keypoints, 'left')
                right_knee_tracking = self.squat_knee_tracking_score(keypoints, 'right')
                
                # Only use valid (non-None) scores and be extremely strict
                valid_scores = [s for s in [squat_form_score, knee_score, spine_score, core_score] if s is not None]
                knee_tracking_scores = [s for s in [left_knee_tracking, right_knee_tracking] if s is not None]
                
                if len(valid_scores) < 2:  # Need at least 2 reliable measurements
                    combined_form_score = 5  # Very poor form if can't track properly
                else:
                    combined_form_score = int(sum(valid_scores) / len(valid_scores))
                    
                    # CRITICAL: Heavily penalize if knees go over toes (safety issue)
                    if knee_tracking_scores:
                        worst_knee_tracking = min(knee_tracking_scores)
                        if worst_knee_tracking < 20:  # Poor knee tracking
                            combined_form_score = int(combined_form_score * 0.2)  # 80% penalty for knee over toes
                        elif worst_knee_tracking < 40:  # Moderate knee issues
                            combined_form_score = int(combined_form_score * 0.5)  # 50% penalty
                    
                    # Additional penalty for wild movements
                    if len(valid_scores) < 3:
                        combined_form_score = int(combined_form_score * 0.5)  # 50% penalty for missing data
                    
                    # Check for movement stability
                    if len(self.angle_history) >= 3:
                        angle_variance = np.var(self.angle_history)
                        if angle_variance > 500:  # High variance indicates erratic movement
                            combined_form_score = int(combined_form_score * 0.3)  # 70% penalty
                        elif angle_variance > 200:  # Moderate variance
                            combined_form_score = int(combined_form_score * 0.6)  # 40% penalty
                
                # Ensure minimum score constraints
                if combined_form_score < 10:
                    combined_form_score = max(5, combined_form_score)
                
                if progression_result is True:
                    # Check hold requirements for squats - must hold at bottom
                    hold_met, hold_duration, position_stable = self.check_hold_requirements(
                        self.current_exercise, smoothed_angle, keypoints, current_time)
                    
                    if hold_met:
                        rep_completed = True
                        self.last_rep_time = current_time
                        form_score = combined_form_score
                        self.state = 'down'  # Reset to down state
                        self.reset_hold_tracking()
                        # Force state transition by resetting progression
                        self.current_progression = 0
                        self.progression_direction = 'down'
                        self.progressive_states = []
                        self.speak(f"Squat depth held for {hold_duration:.1f} seconds! Rep completed")
                    else:
                        # Still need to hold bottom position longer
                        form_score = max(30, combined_form_score)  # Penalize for incomplete hold
                        
                elif progression_result is False:
                    self.state = 'down'
                    form_score = 10
                    self.reset_hold_tracking()
                else:
                    # Check hold requirements during movement
                    hold_met, hold_duration, position_stable = self.check_hold_requirements(
                        self.current_exercise, smoothed_angle, keypoints, current_time)
                    
                    # Standard squat logic with strict form validation AND HOLD REQUIREMENTS
                    if self.state == 'down' and smoothed_angle > exercise['up_threshold'] - 15:
                        self.state = 'up'
                    elif self.state == 'up' and smoothed_angle < exercise['down_threshold'] + 15:
                        if current_time - self.last_rep_time > self.min_rep_time and combined_form_score >= 50 and hold_met:  # HOLD REQUIREMENT ADDED
                            rep_completed = True
                            self.last_rep_time = current_time
                            form_score = combined_form_score
                            self.reset_hold_tracking()
                        self.state = 'down'

            elif exercise_type in ['push_up', 'push_ups']:
                # Enhanced push-up with comprehensive form validation
                core_score = self.core_engagement_score(keypoints)
                shoulder_score = self.shoulder_stability_score(keypoints)
                alignment_score = self.leg_alignment_score(keypoints)
                spine_score = self.spine_straight_score(keypoints)
                plank_score = self.plank_stability_score(keypoints)  # New plank form method
                
                # Only use valid (non-None) scores and be extremely strict
                valid_scores = [s for s in [core_score, shoulder_score, alignment_score, spine_score, plank_score] if s is not None]
                
                if len(valid_scores) < 2:  # Need at least 2 reliable measurements
                    combined_form_score = 5  # Very poor form if can't track properly
                else:
                    combined_form_score = int(sum(valid_scores) / len(valid_scores))
                    # Additional penalty for wild movements
                    if len(valid_scores) < 3:
                        combined_form_score = int(combined_form_score * 0.5)  # 50% penalty for missing data
                    
                    # Check for movement stability
                    if len(self.angle_history) >= 3:
                        angle_variance = np.var(self.angle_history)
                        if angle_variance > 500:  # High variance indicates erratic movement
                            combined_form_score = int(combined_form_score * 0.3)  # 70% penalty
                        elif angle_variance > 200:  # Moderate variance
                            combined_form_score = int(combined_form_score * 0.6)  # 40% penalty
                
                # Ensure minimum score constraints
                if combined_form_score < 10:
                    combined_form_score = max(5, combined_form_score)
                
                if progression_result is True:
                    rep_completed = True
                    self.last_rep_time = current_time
                    form_score = combined_form_score
                    self.state = 'down'
                elif progression_result is False:
                    self.state = 'down'
                    form_score = 10
                else:
                    # Standard push-up logic with form validation
                    if self.state == 'down' and smoothed_angle > exercise['up_threshold'] - 15:
                        self.state = 'up'
                    elif self.state == 'up' and smoothed_angle < exercise['down_threshold'] + 15:
                        if current_time - self.last_rep_time > self.min_rep_time and combined_form_score >= 40:
                            rep_completed = True
                            self.last_rep_time = current_time
                            form_score = combined_form_score
                        self.state = 'down'

            elif exercise_type in ['lunge', 'dumbbell_lunge', 'reverse_lunge', 'forward_lunge']:
                # Enhanced lunge with comprehensive form validation
                lunge_score = self.lunge_stability_score(keypoints)  # New lunge method
                knee_score = self.knee_angle_score(keypoints, None, target_angle=90)
                alignment_score = self.leg_alignment_score(keypoints)
                spine_score = self.spine_straight_score(keypoints)
                core_score = self.core_engagement_score(keypoints)
                
                # Only use valid (non-None) scores and be extremely strict
                valid_scores = [s for s in [lunge_score, knee_score, alignment_score, spine_score, core_score] if s is not None]
                
                if len(valid_scores) < 2:  # Need at least 2 reliable measurements
                    combined_form_score = 5  # Very poor form if can't track properly
                else:
                    combined_form_score = int(sum(valid_scores) / len(valid_scores))
                    # Additional penalty for wild movements
                    if len(valid_scores) < 3:
                        combined_form_score = int(combined_form_score * 0.5)  # 50% penalty for missing data
                    
                    # Check for movement stability
                    if len(self.angle_history) >= 3:
                        angle_variance = np.var(self.angle_history)
                        if angle_variance > 500:  # High variance indicates erratic movement
                            combined_form_score = int(combined_form_score * 0.3)  # 70% penalty
                        elif angle_variance > 200:  # Moderate variance
                            combined_form_score = int(combined_form_score * 0.6)  # 40% penalty
                
                # Ensure minimum score constraints
                if combined_form_score < 10:
                    combined_form_score = max(5, combined_form_score)
                
                if progression_result is True:
                    rep_completed = True
                    self.last_rep_time = current_time
                    form_score = combined_form_score
                    self.state = 'down'
                elif progression_result is False:
                    self.state = 'down'
                    form_score = 10
                else:
                    # Standard lunge logic with form validation
                    if self.state == 'down' and smoothed_angle > exercise['up_threshold'] - 15:
                        self.state = 'up'
                    elif self.state == 'up' and smoothed_angle < exercise['down_threshold'] + 15:
                        if current_time - self.last_rep_time > self.min_rep_time and combined_form_score >= 40:
                            rep_completed = True
                            self.last_rep_time = current_time
                            form_score = combined_form_score
                        self.state = 'down'

            elif exercise_type in ['plank', 'side_plank']:
                # Enhanced plank with comprehensive stability validation
                plank_score = self.plank_stability_score(keypoints)  # New plank method
                core_score = self.core_engagement_score(keypoints)
                shoulder_score = self.shoulder_stability_score(keypoints)
                spine_score = self.spine_straight_score(keypoints)
                
                # Only use valid (non-None) scores and be extremely strict
                valid_scores = [s for s in [plank_score, core_score, shoulder_score, spine_score] if s is not None]
                
                if len(valid_scores) < 2:  # Need at least 2 reliable measurements
                    combined_form_score = 5  # Very poor form if can't track properly
                else:
                    combined_form_score = int(sum(valid_scores) / len(valid_scores))
                    # Additional penalty for wild movements
                    if len(valid_scores) < 3:
                        combined_form_score = int(combined_form_score * 0.5)  # 50% penalty for missing data
                    
                    # Check for movement stability (critical for planks)
                    if len(self.angle_history) >= 3:
                        angle_variance = np.var(self.angle_history)
                        if angle_variance > 300:  # Lower threshold for planks (should be stable)
                            combined_form_score = int(combined_form_score * 0.2)  # 80% penalty for instability
                        elif angle_variance > 100:  # Moderate variance
                            combined_form_score = int(combined_form_score * 0.5)  # 50% penalty
                
                # Ensure minimum score constraints
                if combined_form_score < 10:
                    combined_form_score = max(5, combined_form_score)
                
                # For planks, track time holding position rather than reps
                if self.state == 'down' and smoothed_angle > exercise['up_threshold'] - 10:
                    self.state = 'up'
                elif self.state == 'up' and smoothed_angle < exercise['down_threshold'] + 10:
                    if current_time - self.last_rep_time > 2.0 and combined_form_score >= 60:  # Hold for 2+ seconds with good form
                        rep_completed = True
                        self.last_rep_time = current_time
                        form_score = combined_form_score
                    self.state = 'down'

            else:
                # Enhanced form validation for ALL other exercises - NO MORE BASIC FALLBACKS
                spine_score = self.spine_straight_score(keypoints)
                core_score = self.core_engagement_score(keypoints)
                shoulder_score = self.shoulder_stability_score(keypoints)
                alignment_score = self.leg_alignment_score(keypoints)
                
                # Only use valid (non-None) scores and be extremely strict
                valid_scores = [s for s in [spine_score, core_score, shoulder_score, alignment_score] if s is not None]
                
                if len(valid_scores) < 2:  # Need at least 2 reliable measurements
                    combined_form_score = 5  # Very poor form if can't track properly
                else:
                    combined_form_score = int(sum(valid_scores) / len(valid_scores))
                    # Additional penalty for wild movements
                    if len(valid_scores) < 3:
                        combined_form_score = int(combined_form_score * 0.5)  # 50% penalty for missing data
                    
                    # Check for movement stability
                    if len(self.angle_history) >= 3:
                        angle_variance = np.var(self.angle_history)
                        if angle_variance > 500:  # High variance indicates erratic movement
                            combined_form_score = int(combined_form_score * 0.3)  # 70% penalty
                        elif angle_variance > 200:  # Moderate variance
                            combined_form_score = int(combined_form_score * 0.6)  # 40% penalty
                
                # Ensure minimum score constraints
                if combined_form_score < 10:
                    combined_form_score = max(5, combined_form_score)
                
                # For all other exercises: down = small angle, up = large angle with STRICT form requirements
                
                if progression_result is True:
                    # Progressive rep completed - complete immediately (NO HOLD)
                    rep_completed = True
                    self.last_rep_time = current_time
                    form_score = combined_form_score
                    self.state = 'down'
                    self.current_progression = 0
                    self.progression_direction = 'down'
                    self.progressive_states = []
                elif progression_result is False:
                    self.state = 'down'
                    form_score = 10
                    self.reset_hold_tracking()
                elif self.state == 'down' and smoothed_angle > exercise['up_threshold'] - 15:
                    self.state = 'up'
                elif self.state == 'up' and smoothed_angle < exercise['down_threshold'] + 15:
                    if current_time - self.last_rep_time > self.min_rep_time and combined_form_score >= 50 and hold_met:  # HOLD REQUIREMENT FOR ALL
                        rep_completed = True
                        self.last_rep_time = current_time
                        form_score = combined_form_score
                        self.reset_hold_tracking()
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
        
        # Target hold angle/position
        if hasattr(self, 'current_exercise') and self.current_exercise:
            exercise_def = self.exercises.get(self.current_exercise, {})
            hold_reqs = exercise_def.get('hold_requirements', {})
            target_angle = hold_reqs.get('angle_requirement', hold_reqs.get('height_requirement', None))
            if target_angle:
                cv2.putText(frame, f"Hold Target:", 
                           (text_x, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"{target_angle:.0f}", 
                           (text_x, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Movement state
        cv2.putText(frame, f"State:", 
                   (text_x, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{self.state}", 
                   (text_x, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Form score
        if hasattr(self, 'last_form_score') and self.last_form_score is not None:
            cv2.putText(frame, f"Form Score:", 
                       (text_x, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            color = (0, 255, 0) if self.last_form_score >= 80 else (0, 255, 255) if self.last_form_score >= 60 else (0, 0, 255)
            cv2.putText(frame, f"{self.last_form_score}%", 
                       (text_x, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Hold duration and requirements
        if hasattr(self, 'current_hold_duration'):
            cv2.putText(frame, f"Hold Duration:", 
                       (text_x, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Get required hold time for current exercise
            exercise_def = self.exercises.get(self.current_exercise, {})
            hold_reqs = exercise_def.get('hold_requirements', {})
            required_time = hold_reqs.get('top_hold', hold_reqs.get('bottom_hold', hold_reqs.get('position_hold', 2.0)))
            
            # Color code based on progress
            if self.current_hold_duration >= required_time:
                hold_color = (0, 255, 0)  # Green - requirement met
                status = "✓ HOLD COMPLETE"
            elif self.target_position_stable:
                hold_color = (0, 255, 255)  # Yellow - holding but not complete
                status = f"{self.current_hold_duration:.1f}s / {required_time:.1f}s"
            else:
                hold_color = (0, 0, 255)  # Red - not in position
                status = "Position unstable"
            
            cv2.putText(frame, status, 
                       (text_x, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hold_color, 2)
        
        # Current tracking side for bilateral exercises
        if hasattr(self, 'current_tracking_side') and self.current_tracking_side:
            cv2.putText(frame, f"Tracking Side:", 
                       (text_x, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            side_color = (255, 0, 255) if self.current_tracking_side == 'left' else (0, 255, 255)
            cv2.putText(frame, f"{self.current_tracking_side.upper()}", 
                       (text_x, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.6, side_color, 2)
    
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
            for key, exercise in list(tracker.exercises.items())[10:20]:  # Show next 10
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
