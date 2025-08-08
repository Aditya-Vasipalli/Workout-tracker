"""
Enhanced Workout Tracker with 25+ Exercise Support
Real-time pose detection and form analysis using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import pyttsx3
from datetime import datetime
from exercise_tracker import ExerciseTracker

class EnhancedWorkoutTracker:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,  # Lower for partial body visibility  
            min_tracking_confidence=0.5,   # Lower for leg exercises
            model_complexity=1             # Simpler model for better performance
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize TTS with female voice
        self.tts_engine = pyttsx3.init()
        self.setup_female_voice()
        
        # Exercise tracker
        self.exercise_tracker = ExerciseTracker()
        
        # Workout state
        self.current_exercise = None
        self.current_set = 0
        self.target_reps = 0
        self.workout_log = []
        self.set_start_time = None
        
        # Supported exercises
        self.supported_exercises = {
            # Leg exercises
            'donkey_kicks': 'Donkey Kicks',
            'fire_hydrants': 'Fire Hydrants', 
            'glute_bridges': 'Glute Bridges',
            'hip_thrusts': 'Hip Thrusts',
            'plie_squats': 'Plie Squats',
            'lunges': 'Lunges',
            'courtsey_lunges': 'Courtesy Lunges',
            'side_leg_raises': 'Side Leg Raises',
            'calf_raises': 'Calf Raises',
            'squat': 'Regular Squats',
            
            # Core exercises
            'v_hold': 'V-Hold',
            'boat_hold_center': 'Boat Hold (Center)',
            'boat_hold_left': 'Boat Hold (Left)',
            'boat_hold_right': 'Boat Hold (Right)',
            'planks': 'Planks',
            'crunches': 'Crunches',
            'leg_raises': 'Leg Raises',
            'flutter_kicks': 'Flutter Kicks',
            'russian_twists': 'Russian Twists',
            
            # Upper body exercises
            'push_ups': 'Push-Ups',
            'shoulder_press': 'Shoulder Press',
            'arnold_press': 'Arnold Press',
            'bent_over_rows': 'Bent-Over Rows',
            'dumbbell_pullover': 'Dumbbell Pullover',
            'bicep_curl': 'Bicep Curls',
            
            # Specialized exercises
            'pilates_clamshell': 'Pilates Clamshell',
            
            # Stretches (time-based)
            'butterfly_stretch': 'Butterfly Stretch',
            'cobra_stretch': 'Cobra Stretch',
            'frog_stretch': 'Frog Stretch'
        }
        
        # Time-based exercises (measured in seconds rather than reps)
        self.time_based_exercises = {
            'v_hold', 'boat_hold_center', 'boat_hold_left', 'boat_hold_right',
            'planks', 'butterfly_stretch', 'cobra_stretch', 'frog_stretch'
        }
    
    def setup_female_voice(self):
        """Set up female voice for TTS"""
        try:
            voices = self.tts_engine.getProperty('voices')
            print(f"🔍 Available voices: {len(voices)} found")
            
            # Look for female voice with better detection
            female_voice = None
            for i, voice in enumerate(voices):
                print(f"  Voice {i}: {voice.name} (ID: {voice.id})")
                name_lower = voice.name.lower()
                id_lower = voice.id.lower()
                
                # Enhanced female voice detection
                female_keywords = ['female', 'woman', 'zira', 'hazel', 'susan', 'samantha', 'anna', 'emily', 'eva']
                if any(keyword in name_lower or keyword in id_lower for keyword in female_keywords):
                    female_voice = voice
                    print(f"  🎯 Selected female voice: {voice.name}")
                    break
                elif i == 1 and len(voices) > 1:  # Often the second voice is female on Windows
                    female_voice = voice
                    print(f"  🎯 Using voice 1 (often female): {voice.name}")
            
            if female_voice:
                self.tts_engine.setProperty('voice', female_voice.id)
                print(f"✓ Female voice configured: {female_voice.name}")
                
                # Test the voice to confirm it's working
                print("Testing voice...")
                self.tts_engine.say("Voice test - ready to workout")
                self.tts_engine.runAndWait()
            else:
                print("  ⚠️  No voice found, using default")
                
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"TTS setup warning: {e}")
    
    def speak(self, text, blocking=True):
        """Text-to-speech output with optional non-blocking mode"""
        try:
            print(f"🗣️  {text}")
            if hasattr(self, 'tts_engine') and self.tts_engine:
                if blocking:
                    # Blocking mode - wait for speech to complete
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                else:
                    # Non-blocking mode - just queue the speech
                    import threading
                    def speak_async():
                        try:
                            self.tts_engine.say(text)
                            self.tts_engine.runAndWait()
                        except:
                            pass
                    thread = threading.Thread(target=speak_async)
                    thread.daemon = True
                    thread.start()
            else:
                print("⚠️ TTS engine not available")
                print(f">>> {text} <<<")  # Visual fallback
        except Exception as e:
            print(f"TTS error: {e}")
            print(f">>> {text} <<<")  # Visual fallback
    
    def show_supported_exercises(self):
        """Display all supported exercises"""
        print("\n🏋️‍♀️ SUPPORTED EXERCISES")
        print("=" * 60)
        
        categories = {
            "Leg Exercises": ['donkey_kicks', 'fire_hydrants', 'glute_bridges', 'hip_thrusts', 
                             'plie_squats', 'lunges', 'courtsey_lunges', 'side_leg_raises', 
                             'calf_raises', 'squat'],
            "Core Exercises": ['v_hold', 'boat_hold_center', 'boat_hold_left', 'boat_hold_right',
                              'planks', 'crunches', 'leg_raises', 'flutter_kicks', 'russian_twists'],
            "Upper Body": ['push_ups', 'shoulder_press', 'arnold_press', 'bent_over_rows', 
                          'dumbbell_pullover', 'bicep_curl'],
            "Specialized": ['pilates_clamshell'],
            "Stretches": ['butterfly_stretch', 'cobra_stretch', 'frog_stretch']
        }
        
        for category, exercises in categories.items():
            print(f"\n📂 {category}:")
            for exercise in exercises:
                display_name = self.supported_exercises.get(exercise, exercise)
                time_note = " (time-based)" if exercise in self.time_based_exercises else ""
                print(f"   • {exercise} - {display_name}{time_note}")
    
    def get_workout_plan(self):
        """Get workout plan from user"""
        print("\n🏋️‍♀️ CREATE YOUR WORKOUT")
        print("=" * 50)
        
        self.show_supported_exercises()
        
        workout_plan = []
        print(f"\nEnter your workout plan (press Enter when done):")
        
        while True:
            print("\n" + "-" * 30)
            exercise = input("Exercise name: ").strip().lower().replace(' ', '_')
            
            if not exercise:
                break
                
            if exercise not in self.supported_exercises:
                print(f"❌ '{exercise}' not supported. Please use one from the list above.")
                continue
            
            try:
                sets = int(input("Number of sets: "))
                
                if exercise in self.time_based_exercises:
                    reps = int(input("Duration in seconds: "))
                    print(f"✅ Added: {self.supported_exercises[exercise]} - {sets} sets x {reps} seconds")
                else:
                    reps = int(input("Reps per set: "))
                    print(f"✅ Added: {self.supported_exercises[exercise]} - {sets} sets x {reps} reps")
                
                workout_plan.append({
                    "name": exercise,
                    "sets": sets,
                    "reps": reps
                })
                
            except ValueError:
                print("❌ Please enter valid numbers")
                continue
        
        return workout_plan
    
    def run_workout(self, workout_plan):
        """Execute the workout plan"""
        if not workout_plan:
            print("❌ No workout plan provided")
            return
        
        print("\n🚀 STARTING WORKOUT")
        print("=" * 50)
        
        total_exercises = len(workout_plan)
        
        for exercise_idx, exercise in enumerate(workout_plan, 1):
            exercise_name = exercise["name"]
            total_sets = exercise["sets"]
            target_reps = exercise["reps"]
            
            display_name = self.supported_exercises.get(exercise_name, exercise_name)
            unit = "seconds" if exercise_name in self.time_based_exercises else "reps"
            
            print(f"\n🎯 Exercise {exercise_idx}/{total_exercises}: {display_name}")
            print(f"Target: {total_sets} sets x {target_reps} {unit}")
            
            self.speak(f"Starting {display_name}")
            
            for set_num in range(1, total_sets + 1):
                self.current_exercise = exercise_name
                self.current_set = set_num
                self.target_reps = target_reps
                
                print(f"\n🏋️‍♀️ Set {set_num}/{total_sets}")
                input("Press Enter when ready...")
                
                self.speak(f"Start Set {set_num} of {display_name}")
                
                # Run exercise detection
                reps_completed, avg_form_score = self.run_exercise_detection()
                
                # Log the set
                set_data = {
                    "exercise": exercise_name,
                    "set": set_num,
                    "reps_done": reps_completed,
                    "target_reps": target_reps,
                    "form_score": avg_form_score,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                self.workout_log.append(set_data)
                
                # Display results
                unit_display = "seconds" if exercise_name in self.time_based_exercises else "reps"
                print(f"✅ {display_name} - Set {set_num}: {reps_completed} {unit_display} complete")
                print(f"🧠 Form Score: {avg_form_score}%")
                
                if reps_completed >= target_reps:
                    self.speak("Set complete. Good job!")
                else:
                    self.speak(f"Set complete. You did {reps_completed} {unit_display}")
                
                # Rest between sets
                if set_num < total_sets:
                    print("😴 Rest between sets...")
                    time.sleep(2)
        
        self.complete_workout()
    
    def run_exercise_detection(self):
        """Run pose detection for current exercise"""
        # Add initialization period BEFORE opening camera
        import time
        
        print(f"📹 Preparing camera for {self.current_exercise}")
        print(f"Target: {self.target_reps} {'seconds' if self.current_exercise in self.time_based_exercises else 'reps'}")
        
        # Countdown before starting
        initialization_time = 3
        print(f"⏰ Get in position! Starting in {initialization_time} seconds...")
        for i in range(initialization_time, 0, -1):
            print(f"   Starting in {i}...")
            time.sleep(1)
        print("🚀 GO! Start your exercise!")
        self.speak("Go!")
        
        # NOW open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return 0, 0
        
        # Reset exercise tracker for new set
        self.exercise_tracker.reset_exercise()
        
        # Special state for certain exercises
        if 'twist' in self.current_exercise:
            self.exercise_tracker.state = 'center'
        elif 'clamshell' in self.current_exercise:
            self.exercise_tracker.state = 'closed'
        
        print("Position yourself in view and press 'q' to quit, 's' to complete set")
        
        # Reset tracker state 
        self.exercise_tracker.reset_exercise()
        
        # Define unit for display
        unit = "seconds" if self.current_exercise in self.time_based_exercises else "reps"
        
        last_rep_announced = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Draw pose landmarks
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                # Add pose detection status to frame
                cv2.putText(frame, "✓ Pose Detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Track exercise
                rep_completed, form_score = self.exercise_tracker.track_exercise(
                    self.current_exercise, results.pose_landmarks)
                
                # Get and display exercise metrics
                try:
                    metrics = self.exercise_tracker.get_exercise_metrics()
                    angle_text = f"Angle: {metrics['current_angle']}° (Target: {metrics['target_range'][0]}-{metrics['target_range'][1]}°)"
                    state_text = f"State: {metrics['state']}"
                    
                    # Color-code based on angle range
                    angle_color = (0, 255, 0)  # Green for good
                    if metrics['current_angle'] < metrics['target_range'][0] or metrics['current_angle'] > metrics['target_range'][1]:
                        angle_color = (0, 0, 255)  # Red for bad
                    elif metrics['current_angle'] < metrics['target_range'][0] + 10 or metrics['current_angle'] > metrics['target_range'][1] - 10:
                        angle_color = (0, 255, 255)  # Yellow for close
                    
                    cv2.putText(frame, angle_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, angle_color, 2)
                    cv2.putText(frame, state_text, (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except:
                    pass  # Fallback if metrics not available
                
                if rep_completed:
                    print(f"Rep attempt - Form: {form_score}%")
                    
                    # Only count and announce good form reps
                    if form_score >= 70:  # Raised threshold to reduce sensitivity
                        # Good form - count the rep
                        self.exercise_tracker.rep_count += 1
                        current_rep = self.exercise_tracker.rep_count
                        print(f"✅ Rep {current_rep} completed - Good form!")
                        print(f"🔊 DEBUG: About to speak rep number: '{current_rep}'")
                        self.speak(str(current_rep), blocking=False)  # Non-blocking during camera
                        print(f"🔊 DEBUG: Finished speaking rep number")
                        self.exercise_tracker.form_scores.append(form_score)
                        last_rep_announced = current_rep
                    else:
                        # Bad form - don't count rep, but don't spam feedback
                        if form_score < 50:  # Only give feedback for really bad form
                            print(f"❌ Very bad form ({form_score}%) - Try again")
                            print(f"🔊 DEBUG: About to speak bad form feedback")
                            self.speak("Try again, keep your form", blocking=False)  # Non-blocking during camera
                            print(f"🔊 DEBUG: Finished speaking bad form feedback")
                        else:
                            print(f"⚠️ Poor form ({form_score}%) - improving...")  # Just show, don't speak
                    
                    # Check if target reached (only count good reps)
                    if self.exercise_tracker.rep_count >= self.target_reps:
                        print(f"🎉 Target reached! {self.target_reps} {unit} completed")
                        break
            else:
                # No pose detected
                cv2.putText(frame, "⚠ Position yourself in frame", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Show hips/legs for leg exercises", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Display info on frame
            current_count = self.exercise_tracker.rep_count
            progress_text = f"{current_count}/{self.target_reps} {unit}"
            cv2.putText(frame, progress_text, (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Exercise name
            exercise_name = self.supported_exercises.get(self.current_exercise, self.current_exercise)
            cv2.putText(frame, exercise_name, (10, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Form criteria explanation
            form_tips = {
                'calf_raises': 'Rise on toes, feel calf stretch (5-20° height)',
                'squats': 'Knees behind toes, hips back (70-120° knee angle)',
                'lunges': 'Deep step, both knees 90° (85-110° front knee)',
                'push_ups': 'Straight body, full range (90-160° elbow angle)',
                'bicep_curl': 'Elbow close to body, full range (30-160° elbow angle)',
                'donkey_kicks': 'Keep core tight, kick back controlled'
            }
            
            if self.current_exercise in form_tips:
                cv2.putText(frame, form_tips[self.current_exercise], (10, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Form score feedback (60% threshold)
            cv2.putText(frame, "Good form: ≥60% | Try again: <60%", (10, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
            
            cv2.imshow('Workout Tracker', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Workout interrupted by user.")
                break
            elif key == ord('s'):
                print("Set completed manually.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate average form score
        if self.exercise_tracker.form_scores:
            avg_form = sum(self.exercise_tracker.form_scores) // len(self.exercise_tracker.form_scores)
        else:
            avg_form = 0
        
        return self.exercise_tracker.rep_count, avg_form
    
    def complete_workout(self):
        """Complete workout and save log"""
        print("\n🏁 WORKOUT COMPLETE!")
        print("=" * 50)
        
        # Display summary
        exercise_summaries = {}
        for entry in self.workout_log:
            exercise = entry["exercise"]
            if exercise not in exercise_summaries:
                exercise_summaries[exercise] = {
                    "sets": 0,
                    "total_reps": 0,
                    "form_scores": []
                }
            
            exercise_summaries[exercise]["sets"] += 1
            exercise_summaries[exercise]["total_reps"] += entry["reps_done"]
            exercise_summaries[exercise]["form_scores"].append(entry["form_score"])
        
        print("Summary:")
        for exercise, summary in exercise_summaries.items():
            display_name = self.supported_exercises.get(exercise, exercise)
            avg_form = sum(summary["form_scores"]) // len(summary["form_scores"]) if summary["form_scores"] else 0
            unit = "seconds" if exercise in self.time_based_exercises else "reps"
            
            print(f"- {display_name}: {summary['sets']} sets, {summary['total_reps']} {unit}, Avg Form: {avg_form}%")
        
        # Save to file
        filename = f"workout_session_{datetime.now().strftime('%Y%m%d_%H%M')}.bin"
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.workout_log, f)
            print(f"\n💾 Workout saved to {filename}")
        except Exception as e:
            print(f"❌ Could not save workout: {e}")
        
        self.speak("Workout complete! Great job!")

def read_log(filename=None):
    """Read and display workout log"""
    if filename is None:
        # Look for the most recent workout file
        import glob
        files = glob.glob("workout_session_*.bin")
        if not files:
            print("No workout log files found.")
            return
        filename = max(files)  # Most recent file
    
    try:
        with open(filename, 'rb') as f:
            workout_data = pickle.load(f)
        
        print(f"\n📖 WORKOUT LOG: {filename}")
        print("=" * 60)
        
        current_exercise = None
        for entry in workout_data:
            if entry["exercise"] != current_exercise:
                current_exercise = entry["exercise"]
                display_name = entry.get("display_name", current_exercise.replace('_', ' ').title())
                print(f"\n🏋️‍♀️ {display_name}")
                print("-" * 30)
            
            unit = "seconds" if current_exercise in ['v_hold', 'boat_hold_center', 'boat_hold_left', 
                                                    'boat_hold_right', 'planks', 'butterfly_stretch', 
                                                    'cobra_stretch', 'frog_stretch'] else "reps"
            
            print(f"Set {entry['set']}: {entry['reps_done']}/{entry['target_reps']} {unit} "
                  f"(Form: {entry['form_score']}%) at {entry['timestamp']}")
        
    except FileNotFoundError:
        print(f"Log file {filename} not found.")
    except Exception as e:
        print(f"Error reading log: {e}")

def main():
    """Main workout program"""
    print("🏋️‍♀️ ENHANCED AI WORKOUT TRACKER")
    print("Real-time pose detection with 25+ exercises")
    print("=" * 60)
    
    tracker = EnhancedWorkoutTracker()
    
    # Get workout plan
    workout_plan = tracker.get_workout_plan()
    
    if not workout_plan:
        print("No exercises added. Exiting.")
        return
    
    print(f"\n📋 Your workout plan:")
    for i, exercise in enumerate(workout_plan, 1):
        display_name = tracker.supported_exercises.get(exercise["name"], exercise["name"])
        unit = "seconds" if exercise["name"] in tracker.time_based_exercises else "reps"
        print(f"{i}. {display_name}: {exercise['sets']} sets x {exercise['reps']} {unit}")
    
    if input("\nStart workout? (y/n): ").lower().startswith('y'):
        tracker.run_workout(workout_plan)
    else:
        print("Workout cancelled.")

if __name__ == "__main__":
    main()
