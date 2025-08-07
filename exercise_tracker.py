"""
Comprehensive Exercise Tracking Module
Supports 25+ exercises with MediaPipe pose detection
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time

class ExerciseTracker:
    def __init__(self):
        # Initialize MediaPipe with lower confidence for partial body visibility
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,  # Lower for leg exercises
            min_tracking_confidence=0.5,   # Lower for leg exercises
            model_complexity=1             # Use simpler model for better performance
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Exercise state tracking
        self.state = 'down'  # up/down for most exercises
        self.rep_count = 0
        self.last_rep_time = 0
        self.form_scores = []
        self.min_rep_time = 1.5  # Increased minimum time between reps (1.5 seconds)
        self.state_change_time = 0  # Track when state last changed
        self.min_state_time = 0.5  # Minimum time to stay in a state before changing
        self.current_angle = 0  # Track current angle for display
        self.target_angle_range = (0, 180)  # Target angle range for current exercise
        self.movement_history = []  # Track movement for stability
        self.stable_frames_required = 3  # Frames needed for stable detection
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def get_distance(self, point1, point2):
        """Calculate distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def is_point_stable(self, point, threshold=0.05):
        """Check if a point is relatively stable (not shaking)"""
        # Add current point to history
        self.movement_history.append(point)
        
        # Keep only recent history
        if len(self.movement_history) > 10:
            self.movement_history.pop(0)
        
        # Need minimum frames for stability check
        if len(self.movement_history) < self.stable_frames_required:
            return False
        
        # Check if recent movements are within threshold
        recent_points = self.movement_history[-self.stable_frames_required:]
        for i in range(1, len(recent_points)):
            distance = math.sqrt(
                (recent_points[i][0] - recent_points[i-1][0])**2 + 
                (recent_points[i][1] - recent_points[i-1][1])**2
            )
            if distance > threshold:
                return False
        
        return True
    
    def calculate_form_score(self, angle, target_range, stability_bonus=0):
        """Calculate form score based on angle range and stability"""
        min_angle, max_angle = target_range
        
        # Base score from angle range
        if min_angle <= angle <= max_angle:
            angle_score = 100
        else:
            # Penalize based on how far outside the range
            if angle < min_angle:
                angle_score = max(0, 100 - (min_angle - angle) * 2)
            else:
                angle_score = max(0, 100 - (angle - max_angle) * 2)
        
        # Add stability bonus
        total_score = min(100, angle_score + stability_bonus)
        
        return int(total_score)
    
    # =================================================================
    # LEG EXERCISES
    # =================================================================
    
    def track_donkey_kicks(self, pose_landmarks):
        """Track donkey kicks - leg raises from hands and knees position"""
        try:
            # Get relevant landmarks
            left_hip = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
            
            # Calculate hip-knee-ankle angle
            angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            
            # State machine for rep counting
            if self.state == 'down' and angle > 140:  # Leg extended back
                self.state = 'up'
            elif self.state == 'up' and angle < 90:  # Leg brought down
                if self.is_valid_form_donkey_kicks(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(angle, (80, 160))
                    self.form_scores.append(form_score)
                    self.last_rep_time = time.time()
                    return True, form_score
                self.state = 'down'
                
            return False, 0
            
        except Exception as e:
            return False, 0
    
    def is_valid_form_donkey_kicks(self, pose_landmarks):
        """Validate donkey kick form"""
        try:
            # Check if person is in quadruped position (hands and knees)
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            
            # Check if torso is roughly horizontal
            torso_angle = abs(left_shoulder.y - left_hip.y)
            
            return torso_angle < 0.2  # Reasonable threshold for horizontal position
            
        except Exception:
            return False
    
    def track_fire_hydrants(self, pose_landmarks):
        """Track fire hydrants - lateral leg raises from quadruped"""
        try:
            # Get relevant landmarks
            left_hip = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            right_hip = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y]
            
            # Calculate lateral distance of knee from hip line
            hip_distance = self.get_distance(left_knee, left_hip)
            
            # State machine for rep counting
            if self.state == 'down' and hip_distance > 0.3:  # Leg lifted out
                self.state = 'up'
            elif self.state == 'up' and hip_distance < 0.15:  # Leg brought back
                if self.is_valid_form_fire_hydrants(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(hip_distance * 100, (15, 35))
                    self.form_scores.append(form_score)
                    return True, form_score
                self.state = 'down'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_fire_hydrants(self, pose_landmarks):
        """Validate fire hydrant form"""
        return self.is_valid_form_donkey_kicks(pose_landmarks)  # Same quadruped check
    
    def track_glute_bridges(self, pose_landmarks):
        """Track glute bridges - hip thrusts from lying position"""
        try:
            # Get relevant landmarks
            left_shoulder = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                           pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_hip = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            
            # Calculate shoulder-hip-knee angle
            angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            
            # State machine for rep counting
            if self.state == 'down' and angle > 160:  # Hips raised
                self.state = 'up'
            elif self.state == 'up' and angle < 130:  # Hips lowered
                if self.is_valid_form_glute_bridges(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(angle, (130, 170))
                    self.form_scores.append(form_score)
                    return True, form_score
                self.state = 'down'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_glute_bridges(self, pose_landmarks):
        """Validate glute bridge form"""
        try:
            # Check if shoulders are relatively stable (lying down)
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Shoulders should be at similar height (lying down)
            shoulder_level = abs(left_shoulder.y - right_shoulder.y)
            
            return shoulder_level < 0.1
            
        except Exception:
            return False
    
    def track_hip_thrusts(self, pose_landmarks):
        """Track hip thrusts - similar to glute bridges but with back support"""
        # Hip thrusts are very similar to glute bridges in movement pattern
        return self.track_glute_bridges(pose_landmarks)
    
    def is_valid_form_hip_thrusts(self, pose_landmarks):
        """Validate hip thrust form"""
        return self.is_valid_form_glute_bridges(pose_landmarks)
    
    def track_plie_squats(self, pose_landmarks):
        """Track plie squats - wide stance squats"""
        try:
            # Get relevant landmarks
            left_hip = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
            
            # Calculate hip-knee-ankle angle
            angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            
            # State machine for rep counting
            if self.state == 'up' and angle < 90:  # Squatting down
                self.state = 'down'
            elif self.state == 'down' and angle > 160:  # Standing up
                if self.is_valid_form_plie_squats(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(angle, (80, 170))
                    self.form_scores.append(form_score)
                    return True, form_score
                self.state = 'up'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_plie_squats(self, pose_landmarks):
        """Validate plie squat form"""
        try:
            # Check stance width
            left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            stance_width = abs(left_ankle.x - right_ankle.x)
            
            # Plie squats should have wider stance than regular squats
            return stance_width > 0.3
            
        except Exception:
            return False
    
    def track_lunges(self, pose_landmarks):
        """Track lunges - alternating leg forward"""
        try:
            # Get relevant landmarks
            left_hip = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
            
            # Calculate front leg angle
            angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            
            # State machine for rep counting
            if self.state == 'up' and angle < 100:  # Lunging down
                self.state = 'down'
            elif self.state == 'down' and angle > 160:  # Standing up
                if self.is_valid_form_lunges(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(angle, (90, 170))
                    self.form_scores.append(form_score)
                    return True, form_score
                self.state = 'up'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_lunges(self, pose_landmarks):
        """Validate lunge form"""
        try:
            # Check if torso is upright
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            
            torso_angle = abs(left_shoulder.x - left_hip.x)
            
            return torso_angle < 0.15  # Relatively upright torso
            
        except Exception:
            return False
    
    def track_courtsey_lunges(self, pose_landmarks):
        """Track courtsey lunges - diagonal back lunge"""
        # Similar to regular lunges but with diagonal movement
        return self.track_lunges(pose_landmarks)
    
    def is_valid_form_courtsey_lunges(self, pose_landmarks):
        """Validate courtsey lunge form"""
        return self.is_valid_form_lunges(pose_landmarks)
    
    def track_side_leg_raises(self, pose_landmarks):
        """Track side leg raises - lateral leg lifts"""
        try:
            # Get relevant landmarks
            left_hip = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            left_ankle = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
            right_hip = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP].y]
            
            # Calculate lateral distance
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            leg_distance = abs(left_ankle[0] - hip_center_x)
            
            # State machine for rep counting
            if self.state == 'down' and leg_distance > 0.3:  # Leg raised
                self.state = 'up'
            elif self.state == 'up' and leg_distance < 0.1:  # Leg lowered
                if self.is_valid_form_side_leg_raises(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(leg_distance * 100, (10, 40))
                    self.form_scores.append(form_score)
                    return True, form_score
                self.state = 'down'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_side_leg_raises(self, pose_landmarks):
        """Validate side leg raise form"""
        try:
            # Check if person is standing upright
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            
            # Should be relatively vertical
            torso_angle = abs(left_shoulder.x - left_hip.x)
            
            return torso_angle < 0.2
            
        except Exception:
            return False
    
    def track_calf_raises(self, pose_landmarks):
        """Track calf raises - rising on toes"""
        try:
            current_time = time.time()
            
            # Get relevant landmarks
            left_knee = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
            left_heel = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].y]
            
            # Calculate vertical position change
            ankle_height = left_ankle[1]
            
            # State machine for rep counting (lower y = higher position)
            # More strict thresholds and timing to prevent false positives
            if self.state == 'down' and ankle_height < 0.82:  # Raised up (stricter)
                if current_time - self.state_change_time > 0.3:  # Require 0.3s in previous state
                    self.state = 'up'
                    self.state_change_time = current_time
            elif self.state == 'up' and ankle_height > 0.92:  # Lowered down (stricter)
                if current_time - self.state_change_time > 0.3:  # Require 0.3s in up state
                    if current_time - self.last_rep_time > self.min_rep_time:  # Prevent too fast reps
                        if self.is_valid_form_calf_raises(pose_landmarks):
                            form_score = self.calculate_form_score((1 - ankle_height) * 100, (10, 20))
                            self.last_rep_time = current_time
                            return True, form_score
                    self.state = 'down'
                    self.state_change_time = current_time
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_calf_raises(self, pose_landmarks):
        """Validate calf raise form"""
        try:
            # Check if legs are straight
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            
            # Calculate leg straightness
            leg_alignment = abs((left_hip.x - left_ankle.x) - (left_hip.x - left_knee.x))
            
            return leg_alignment < 0.1
            
        except Exception:
            return False
    
    # =================================================================
    # CORE EXERCISES
    # =================================================================
    
    def track_v_hold(self, pose_landmarks):
        """Track V-hold - seated V position hold"""
        try:
            # Get relevant landmarks
            left_shoulder = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                           pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_hip = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            left_ankle = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
            
            # Calculate torso-leg angle for V-shape
            angle = self.calculate_angle(left_shoulder, left_hip, left_ankle)
            
            # For holds, we count time rather than reps
            if self.is_valid_form_v_hold(pose_landmarks) and 60 <= angle <= 90:
                # This is a valid hold position
                current_time = time.time()
                if current_time - self.last_rep_time > 1:  # Count every second
                    self.rep_count += 1
                    form_score = self.calculate_form_score(angle, (60, 90))
                    self.form_scores.append(form_score)
                    self.last_rep_time = current_time
                    return True, form_score
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_v_hold(self, pose_landmarks):
        """Validate V-hold form"""
        try:
            # Check if person is in seated position with legs raised
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            
            # Legs should be elevated
            return left_ankle.y < left_hip.y
            
        except Exception:
            return False
    
    def track_boat_hold_center(self, pose_landmarks):
        """Track boat hold center position"""
        return self.track_v_hold(pose_landmarks)  # Similar to V-hold
    
    def track_boat_hold_left(self, pose_landmarks):
        """Track boat hold with lean to left"""
        return self.track_v_hold(pose_landmarks)
    
    def track_boat_hold_right(self, pose_landmarks):
        """Track boat hold with lean to right"""
        return self.track_v_hold(pose_landmarks)
    
    def is_valid_form_boat_hold_center(self, pose_landmarks):
        return self.is_valid_form_v_hold(pose_landmarks)
    
    def is_valid_form_boat_hold_left(self, pose_landmarks):
        return self.is_valid_form_v_hold(pose_landmarks)
    
    def is_valid_form_boat_hold_right(self, pose_landmarks):
        return self.is_valid_form_v_hold(pose_landmarks)
    
    def track_planks(self, pose_landmarks):
        """Track plank hold - time-based"""
        try:
            # Get relevant landmarks
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            
            # Check if body is in straight line
            if self.is_valid_form_planks(pose_landmarks):
                current_time = time.time()
                if current_time - self.last_rep_time > 1:  # Count every second
                    self.rep_count += 1
                    # Form score based on body alignment
                    body_alignment = abs(left_shoulder.y - left_hip.y) + abs(left_hip.y - left_ankle.y)
                    form_score = max(50, 100 - int(body_alignment * 200))
                    self.form_scores.append(form_score)
                    self.last_rep_time = current_time
                    return True, form_score
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_planks(self, pose_landmarks):
        """Validate plank form"""
        try:
            # Check if body is relatively straight
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            
            # Calculate body line deviation
            shoulder_hip_diff = abs(left_shoulder.y - left_hip.y)
            hip_ankle_diff = abs(left_hip.y - left_ankle.y)
            
            return shoulder_hip_diff < 0.1 and hip_ankle_diff < 0.1
            
        except Exception:
            return False
    
    def track_crunches(self, pose_landmarks):
        """Track crunches - sit-up motion"""
        try:
            # Get relevant landmarks
            left_shoulder = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                           pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_hip = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            
            # Calculate torso angle
            angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
            
            # State machine for rep counting
            if self.state == 'down' and angle < 130:  # Crunching up
                self.state = 'up'
            elif self.state == 'up' and angle > 150:  # Lowering down
                if self.is_valid_form_crunches(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(angle, (120, 160))
                    self.form_scores.append(form_score)
                    return True, form_score
                self.state = 'down'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_crunches(self, pose_landmarks):
        """Validate crunch form"""
        try:
            # Check if knees are bent (lying position)
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            
            # Knees should be higher than ankles
            return left_knee.y < left_ankle.y
            
        except Exception:
            return False
    
    def track_leg_raises(self, pose_landmarks):
        """Track leg raises - lifting legs from lying position"""
        try:
            # Get relevant landmarks
            left_hip = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
            
            # Calculate leg angle from horizontal
            leg_angle = self.calculate_angle([left_hip[0] + 0.1, left_hip[1]], left_hip, left_ankle)
            
            # State machine for rep counting
            if self.state == 'down' and leg_angle > 70:  # Legs raised
                self.state = 'up'
            elif self.state == 'up' and leg_angle < 30:  # Legs lowered
                if self.is_valid_form_leg_raises(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(leg_angle, (20, 80))
                    self.form_scores.append(form_score)
                    return True, form_score
                self.state = 'down'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_leg_raises(self, pose_landmarks):
        """Validate leg raise form"""
        try:
            # Check if torso is stable (lying down)
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Shoulders should be level
            return abs(left_shoulder.y - right_shoulder.y) < 0.1
            
        except Exception:
            return False
    
    def track_flutter_kicks(self, pose_landmarks):
        """Track flutter kicks - alternating leg movements"""
        try:
            # Get relevant landmarks
            left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            # Track alternating movement
            height_diff = abs(left_ankle.y - right_ankle.y)
            
            if height_diff > 0.1:  # Legs are in different positions
                if self.is_valid_form_flutter_kicks(pose_landmarks):
                    current_time = time.time()
                    if current_time - self.last_rep_time > 0.5:  # Count faster than other exercises
                        self.rep_count += 1
                        form_score = self.calculate_form_score(height_diff * 100, (10, 30))
                        self.form_scores.append(form_score)
                        self.last_rep_time = current_time
                        return True, form_score
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_flutter_kicks(self, pose_landmarks):
        """Validate flutter kick form"""
        return self.is_valid_form_leg_raises(pose_landmarks)  # Similar position
    
    def track_russian_twists(self, pose_landmarks):
        """Track Russian twists - seated torso rotation"""
        try:
            # Get relevant landmarks
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            
            # Calculate torso rotation
            shoulder_line = left_shoulder.x - right_shoulder.x
            rotation = abs(shoulder_line)
            
            # State machine for rep counting
            if self.state == 'center' and rotation > 0.15:  # Twisted to side
                self.state = 'side'
            elif self.state == 'side' and rotation < 0.05:  # Back to center
                if self.is_valid_form_russian_twists(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(rotation * 100, (5, 20))
                    self.form_scores.append(form_score)
                    return True, form_score
                self.state = 'center'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_russian_twists(self, pose_landmarks):
        """Validate Russian twist form"""
        try:
            # Check if person is in seated position
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            
            # Hips should be lower than or equal to ankles (seated)
            return left_hip.y >= left_ankle.y
            
        except Exception:
            return False
    
    # =================================================================
    # UPPER BODY EXERCISES
    # =================================================================
    
    def track_push_ups(self, pose_landmarks):
        """Track push-ups"""
        try:
            # Get relevant landmarks
            left_shoulder = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                           pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_elbow = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_wrist = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
            
            # Calculate arm angle
            angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # State machine for rep counting
            if self.state == 'up' and angle < 90:  # Going down
                self.state = 'down'
            elif self.state == 'down' and angle > 160:  # Pushing up
                if self.is_valid_form_push_ups(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(angle, (80, 170))
                    self.form_scores.append(form_score)
                    return True, form_score
                self.state = 'up'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_push_ups(self, pose_landmarks):
        """Validate push-up form"""
        try:
            # Check if body is in plank position
            return self.is_valid_form_planks(pose_landmarks)
            
        except Exception:
            return False
    
    def track_shoulder_press(self, pose_landmarks):
        """Track shoulder press (overhead press)"""
        try:
            # Get relevant landmarks
            left_shoulder = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                           pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_elbow = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_wrist = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
            
            # Calculate arm angle
            angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # State machine for rep counting
            if self.state == 'down' and angle > 160 and left_wrist[1] < left_shoulder[1]:  # Pressed up
                self.state = 'up'
            elif self.state == 'up' and angle < 100:  # Lowered down
                if self.is_valid_form_shoulder_press(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(angle, (90, 170))
                    self.form_scores.append(form_score)
                    return True, form_score
                self.state = 'down'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_shoulder_press(self, pose_landmarks):
        """Validate shoulder press form"""
        try:
            # Check if person is standing upright
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            
            # Torso should be upright
            return abs(left_shoulder.x - left_hip.x) < 0.15
            
        except Exception:
            return False
    
    def track_arnold_press(self, pose_landmarks):
        """Track Arnold press (rotating shoulder press)"""
        # Arnold press is similar to shoulder press in terms of tracking
        return self.track_shoulder_press(pose_landmarks)
    
    def is_valid_form_arnold_press(self, pose_landmarks):
        """Validate Arnold press form"""
        return self.is_valid_form_shoulder_press(pose_landmarks)
    
    def track_bent_over_rows(self, pose_landmarks):
        """Track bent-over rows"""
        try:
            # Get relevant landmarks
            left_shoulder = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                           pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_elbow = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_wrist = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
            
            # Calculate arm angle
            angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # State machine for rep counting
            if self.state == 'down' and angle < 60:  # Rowing up
                self.state = 'up'
            elif self.state == 'up' and angle > 120:  # Lowering down
                if self.is_valid_form_bent_over_rows(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(angle, (50, 130))
                    self.form_scores.append(form_score)
                    return True, form_score
                self.state = 'down'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_bent_over_rows(self, pose_landmarks):
        """Validate bent-over row form"""
        try:
            # Check if person is bent over
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            
            # Should be leaning forward
            lean_forward = left_shoulder.x - left_hip.x
            return lean_forward > 0.1
            
        except Exception:
            return False
    
    def track_dumbbell_pullover(self, pose_landmarks):
        """Track dumbbell pullover"""
        try:
            # Get relevant landmarks
            left_shoulder = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                           pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_elbow = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_wrist = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
            
            # Calculate arm angle for overhead movement
            angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # State machine for rep counting
            if self.state == 'down' and left_wrist[1] < left_shoulder[1] and angle > 150:  # Arms overhead
                self.state = 'up'
            elif self.state == 'up' and left_wrist[1] > left_shoulder[1]:  # Arms lowered
                if self.is_valid_form_dumbbell_pullover(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(angle, (120, 170))
                    self.form_scores.append(form_score)
                    return True, form_score
                self.state = 'down'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_dumbbell_pullover(self, pose_landmarks):
        """Validate dumbbell pullover form"""
        try:
            # Check if person is lying down
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Shoulders should be level (lying down)
            return abs(left_shoulder.y - right_shoulder.y) < 0.1
            
        except Exception:
            return False
    
    # =================================================================
    # SPECIALIZED EXERCISES
    # =================================================================
    
    def track_pilates_clamshell(self, pose_landmarks):
        """Track Pilates clamshell - side-lying leg opens"""
        try:
            # Get relevant landmarks (assuming side-lying position)
            left_hip = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
            
            # Calculate knee angle
            angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            
            # State machine for rep counting
            if self.state == 'closed' and angle > 100:  # Knee opening
                self.state = 'open'
            elif self.state == 'open' and angle < 80:  # Knee closing
                if self.is_valid_form_pilates_clamshell(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(angle, (70, 110))
                    self.form_scores.append(form_score)
                    return True, form_score
                self.state = 'closed'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_pilates_clamshell(self, pose_landmarks):
        """Validate Pilates clamshell form"""
        try:
            # Check if person is in side-lying position
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            
            # Should be relatively horizontal
            return abs(left_shoulder.y - left_hip.y) < 0.15
            
        except Exception:
            return False
    
    # =================================================================
    # STRETCH EXERCISES (TIME-BASED)
    # =================================================================
    
    def track_butterfly_stretch(self, pose_landmarks):
        """Track butterfly stretch - seated with soles together"""
        try:
            if self.is_valid_form_butterfly_stretch(pose_landmarks):
                current_time = time.time()
                if current_time - self.last_rep_time > 1:  # Count every second
                    self.rep_count += 1
                    form_score = 85  # Base score for holding position
                    self.form_scores.append(form_score)
                    self.last_rep_time = current_time
                    return True, form_score
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_butterfly_stretch(self, pose_landmarks):
        """Validate butterfly stretch form"""
        try:
            # Check if person is seated with feet together
            left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            
            # Feet should be close together and hips lower than knees
            feet_distance = abs(left_ankle.x - right_ankle.x)
            return feet_distance < 0.2 and left_hip.y > left_ankle.y
            
        except Exception:
            return False
    
    def track_cobra_stretch(self, pose_landmarks):
        """Track cobra stretch - prone back extension"""
        try:
            if self.is_valid_form_cobra_stretch(pose_landmarks):
                current_time = time.time()
                if current_time - self.last_rep_time > 1:
                    self.rep_count += 1
                    form_score = 85
                    self.form_scores.append(form_score)
                    self.last_rep_time = current_time
                    return True, form_score
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_cobra_stretch(self, pose_landmarks):
        """Validate cobra stretch form"""
        try:
            # Check if person is in prone position with chest raised
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            
            # Shoulders should be higher than hips
            return left_shoulder.y < left_hip.y
            
        except Exception:
            return False
    
    def track_frog_stretch(self, pose_landmarks):
        """Track frog stretch - wide knee position"""
        try:
            if self.is_valid_form_frog_stretch(pose_landmarks):
                current_time = time.time()
                if current_time - self.last_rep_time > 1:
                    self.rep_count += 1
                    form_score = 85
                    self.form_scores.append(form_score)
                    self.last_rep_time = current_time
                    return True, form_score
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_frog_stretch(self, pose_landmarks):
        """Validate frog stretch form"""
        try:
            # Check wide knee position
            left_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            
            # Knees should be wide apart
            knee_distance = abs(left_knee.x - right_knee.x)
            return knee_distance > 0.4
            
        except Exception:
            return False
    
    # =================================================================
    # MAIN TRACKING FUNCTION
    # =================================================================
    
    def track_exercise(self, exercise_name, pose_landmarks):
        """Main function to track any supported exercise"""
        # Reset state for new exercise type
        if not hasattr(self, 'current_exercise') or self.current_exercise != exercise_name:
            self.current_exercise = exercise_name
            self.state = 'down'  # Default state
            if 'twist' in exercise_name.lower():
                self.state = 'center'
            elif 'clamshell' in exercise_name.lower():
                self.state = 'closed'
        
        # Exercise mapping with angle tracking
        exercise_functions = {
            'bicep_curl': self.track_bicep_curls_with_angles,
            'calf_raises': self.track_calf_raises_with_angles,
            'squats': self.track_squats_with_angles,
            'lunges': self.track_lunges_with_angles,
            'push_ups': self.track_push_ups_with_angles,
            # Add more as needed, fallback to original methods
        }
        
        # Use angle-tracking version if available, otherwise use original
        if exercise_name in exercise_functions:
            return exercise_functions[exercise_name](pose_landmarks)
        else:
            # Fallback to original method
            original_result = self.track_exercise_original(exercise_name, pose_landmarks)
            return original_result
    
    def track_exercise_original(self, exercise_name, pose_landmarks):
        """Original tracking method without angle display"""
        
        # Exercise mapping
        exercise_functions = {
            'donkey_kicks': self.track_donkey_kicks,
            'v_hold': self.track_v_hold,
            'fire_hydrants': self.track_fire_hydrants,
            'arnold_press': self.track_arnold_press,
            'glute_bridges': self.track_glute_bridges,
            'pilates_clamshell': self.track_pilates_clamshell,
            'hip_thrusts': self.track_hip_thrusts,
            'plie_squats': self.track_plie_squats,
            'lunges': self.track_lunges,
            'courtsey_lunges': self.track_courtsey_lunges,
            'side_leg_raises': self.track_side_leg_raises,
            'dumbbell_pullover': self.track_dumbbell_pullover,
            'calf_raises': self.track_calf_raises,
            'boat_hold_left': self.track_boat_hold_left,
            'boat_hold_right': self.track_boat_hold_right,
            'boat_hold_center': self.track_boat_hold_center,
            'push_ups': self.track_push_ups,
            'shoulder_press': self.track_shoulder_press,
            'bent_over_rows': self.track_bent_over_rows,
            'planks': self.track_planks,
            'crunches': self.track_crunches,
            'leg_raises': self.track_leg_raises,
            'flutter_kicks': self.track_flutter_kicks,
            'russian_twists': self.track_russian_twists,
            'butterfly_stretch': self.track_butterfly_stretch,
            'cobra_stretch': self.track_cobra_stretch,
            'frog_stretch': self.track_frog_stretch,
            # Add existing exercises
            'bicep_curl': self.track_bicep_curls,
            'squat': self.track_squats
        }
        
        if exercise_name in exercise_functions:
            return exercise_functions[exercise_name](pose_landmarks)
        else:
            print(f"Exercise '{exercise_name}' not supported yet")
            return False, 0
    
    def track_bicep_curls(self, pose_landmarks):
        """Track bicep curls (from original code)"""
        try:
            # Get coordinates
            left_shoulder = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                           pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_elbow = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_wrist = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
            
            # Calculate angle
            angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            # Rep counting logic
            if self.state == 'down' and angle < 50:
                self.state = 'up'
            elif self.state == 'up' and angle > 160:
                if self.is_valid_form_bicep_curls(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(angle, (40, 170))
                    self.form_scores.append(form_score)
                    return True, form_score
                self.state = 'down'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_bicep_curls(self, pose_landmarks):
        """Validate bicep curl form"""
        try:
            # Check if person is standing upright
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            
            return abs(left_shoulder.x - left_hip.x) < 0.15
            
        except Exception:
            return False
    
    def track_squats(self, pose_landmarks):
        """Track squats (from original code)"""
        try:
            # Get coordinates
            left_hip = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
            
            # Calculate angle
            angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            
            # Rep counting logic
            if self.state == 'up' and angle < 90:
                self.state = 'down'
            elif self.state == 'down' and angle > 160:
                if self.is_valid_form_squats(pose_landmarks):
                    self.rep_count += 1
                    form_score = self.calculate_form_score(angle, (80, 170))
                    self.form_scores.append(form_score)
                    return True, form_score
                self.state = 'up'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def is_valid_form_squats(self, pose_landmarks):
        """Validate squat form"""
        try:
            # Check if knees don't cave inward
            left_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            # Knee tracking - knees shouldn't be closer than ankles
            knee_distance = abs(left_knee.x - right_knee.x)
            ankle_distance = abs(left_ankle.x - right_ankle.x)
            
            return knee_distance >= ankle_distance * 0.8
            
        except Exception:
            return False
    
    def reset_exercise(self):
        """Reset exercise tracking state"""
        self.state = 'down'
        self.rep_count = 0
        self.last_rep_time = 0
        self.form_scores = []
        self.state_change_time = 0
        self.current_angle = 0
        self.target_angle_range = (0, 180)
    
    # =================================================================
    # ANGLE-TRACKING EXERCISE METHODS
    # =================================================================
    
    def track_calf_raises_with_angles(self, pose_landmarks):
        """Track calf raises with angle display"""
        try:
            current_time = time.time()
            
            # Get relevant landmarks
            left_knee = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
            left_heel = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL].y]
            
            # Calculate vertical position and ankle angle
            ankle_height = left_ankle[1]
            # Calculate height change as "angle"
            self.current_angle = abs(left_knee[1] - left_ankle[1]) * 100  # Height difference as "angle"
            self.target_angle_range = (5, 20)  # More realistic range
            
            # Relaxed form validation for calf raises
            is_good_form = (
                ankle_height < 0.90 and  # More lenient height requirement
                self.current_angle >= 5   # Minimum height change
            )
            
            # State machine for rep counting (more lenient thresholds)
            if self.state == 'down' and ankle_height < 0.85:  # Raised up
                if current_time - self.state_change_time > 0.2:  # Faster response
                    self.state = 'up'
                    self.state_change_time = current_time
            elif self.state == 'up' and ankle_height > 0.88:  # Lowered down (more lenient)
                if current_time - self.state_change_time > 0.2:  # Faster response
                    if current_time - self.last_rep_time > self.min_rep_time:
                        if is_good_form:
                            form_score = min(100, max(60, self.current_angle * 8))  # More generous scoring
                            self.last_rep_time = current_time
                            return True, form_score
                        else:
                            # Still give some credit for movement
                            self.last_rep_time = current_time
                            return True, 45  # Better than 30%
                    self.state = 'down'
                    self.state_change_time = current_time
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def track_squats_with_angles(self, pose_landmarks):
        """Track squats with knee angle display and strict timing"""
        try:
            current_time = time.time()
            
            # Get relevant landmarks
            left_hip = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
            
            # Calculate knee angle
            angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            self.current_angle = angle
            self.target_angle_range = (70, 180)  # Target range for squats
            
            # Check stability
            knee_stable = self.is_point_stable(left_knee)
            
            # Much stricter state machine with timing controls
            if self.state == 'up' and angle < 90:  # Going down
                # Must stay in up state for minimum time AND be stable
                if (current_time - self.state_change_time > self.min_state_time and 
                    knee_stable and 
                    self.is_valid_form_squats(pose_landmarks)):
                    self.state = 'down'
                    self.state_change_time = current_time
                    print(f"DEBUG SQUAT: State changed to DOWN, angle: {angle:.1f}°")
                    
            elif self.state == 'down' and angle > 160:  # Standing up
                # Must stay in down state for minimum time AND be stable AND minimum rep time passed
                if (current_time - self.state_change_time > self.min_state_time and 
                    current_time - self.last_rep_time > self.min_rep_time and
                    knee_stable and 
                    self.is_valid_form_squats(pose_landmarks)):
                    
                    # Complete the rep
                    form_score = self.calculate_form_score(angle, self.target_angle_range)
                    self.last_rep_time = current_time
                    self.state = 'up'
                    self.state_change_time = current_time
                    print(f"DEBUG SQUAT: REP COMPLETED! angle: {angle:.1f}°, form: {form_score}%")
                    return True, form_score
                    
            return False, 0
            
        except Exception as e:
            print(f"DEBUG SQUAT ERROR: {e}")
            return False, 0
            left_knee = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
            
            # Calculate hip-knee-ankle angle
            self.current_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            self.target_angle_range = (70, 120)  # Good squat range
            
            # Improved form validation
            is_good_form = (
                70 <= self.current_angle <= 120 and  # Proper squat depth
                abs(left_knee[0] - left_ankle[0]) < 0.15 and  # Knees not too far forward
                left_hip[1] > left_knee[1]  # Hips back
            )
            
            # State machine
            if self.state == 'up' and self.current_angle < 100:  # Going down
                self.state = 'down'
            elif self.state == 'down' and self.current_angle > 140:  # Going up
                if is_good_form:
                    form_score = max(60, min(100, 100 - abs(90 - self.current_angle)))
                    return True, form_score
                else:
                    return True, 35  # Bad form
                self.state = 'up'
                
            return False, 0
            
        except Exception:
            return False, 0
    
    def get_exercise_metrics(self):
        """Get current exercise metrics for display"""
        return {
            'current_angle': round(self.current_angle, 1),
            'target_range': self.target_angle_range,
            'state': self.state,
            'rep_count': self.rep_count
        }
    
    def track_bicep_curls_with_angles(self, pose_landmarks):
        """Track bicep curls with elbow angle display"""
        try:
            current_time = time.time()
            
            # Get landmarks
            left_shoulder = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                           pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_elbow = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_wrist = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
            
            # Calculate angle
            angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            self.current_angle = angle
            self.target_angle_range = (30, 160)
            
            # Check stability
            elbow_stable = self.is_point_stable(left_elbow)
            
            # Strict state machine with timing controls
            if self.state == 'down' and angle < 60:  # Curling up (arm bent)
                if (current_time - self.state_change_time > self.min_state_time and 
                    elbow_stable and 
                    self.is_valid_form_bicep_curls(pose_landmarks)):
                    self.state = 'up'
                    self.state_change_time = current_time
                    print(f"DEBUG BICEP: State changed to UP, angle: {angle:.1f}°")
                    
            elif self.state == 'up' and angle > 140:  # Lowering down (arm straight)
                if (current_time - self.state_change_time > self.min_state_time and 
                    current_time - self.last_rep_time > self.min_rep_time and
                    elbow_stable and 
                    self.is_valid_form_bicep_curls(pose_landmarks)):
                    
                    # Complete the rep
                    form_score = self.calculate_form_score(angle, self.target_angle_range)
                    self.last_rep_time = current_time
                    self.state = 'down'
                    self.state_change_time = current_time
                    print(f"DEBUG BICEP: REP COMPLETED! angle: {angle:.1f}°, form: {form_score}%")
                    return True, form_score
                    
            return False, 0
            
        except Exception as e:
            print(f"DEBUG BICEP ERROR: {e}")
            return False, 0
    
    def track_push_ups_with_angles(self, pose_landmarks):
        """Track push-ups with elbow angle display"""
        try:
            left_shoulder = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                           pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_elbow = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_wrist = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST].y]
            
            self.current_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            self.target_angle_range = (90, 160)
            
            is_good_form = (90 <= self.current_angle <= 160 and 
                           abs(left_shoulder[1] - left_wrist[1]) < 0.2)
            
            if self.state == 'up' and self.current_angle < 120:
                self.state = 'down'
            elif self.state == 'down' and self.current_angle > 150:
                if is_good_form:
                    form_score = max(60, min(100, 100 - abs(120 - self.current_angle)))
                    return True, form_score
                else:
                    return True, 25
                self.state = 'up'
            return False, 0
        except Exception:
            return False, 0
    
    def track_lunges_with_angles(self, pose_landmarks):
        """Track lunges with knee angle display"""
        try:
            left_hip = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                       pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP].y]
            left_knee = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                        pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE].y]
            left_ankle = [pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                         pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y]
            
            self.current_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            self.target_angle_range = (85, 110)
            
            is_good_form = (85 <= self.current_angle <= 110 and 
                           left_knee[0] < left_ankle[0] + 0.1)
            
            if self.state == 'up' and self.current_angle < 110:
                self.state = 'down'
            elif self.state == 'down' and self.current_angle > 140:
                if is_good_form:
                    form_score = max(60, min(100, 100 - abs(95 - self.current_angle)))
                    return True, form_score
                else:
                    return True, 30
                self.state = 'up'
            return False, 0
        except Exception:
            return False, 0


