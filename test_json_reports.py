#!/usr/bin/env python3
"""
Test the JSON workout report functionality
"""

import sys
import os
import json
from datetime import datetime

# Import the workout tracker
sys.path.append('.')
from movenet_tracker import MoveNetWorkoutTracker

def create_test_json_report():
    """Create a test JSON workout report"""
    tracker = MoveNetWorkoutTracker()
    
    # Start a test session
    tracker.start_workout_session("Test Glute Workout JSON")
    
    # Simulate first exercise: Glute Bridge
    tracker.start_exercise_tracking("glute_bridge", 15, 3)
    
    # Simulate 3 sets of 15 reps each
    for set_num in range(3):
        for rep in range(15):
            form_score = 91 + (rep % 7)  # Varying form scores 91-97%
            tracker.log_rep_completion(form_score)
        tracker.complete_exercise_set()
    
    tracker.finish_exercise_tracking()
    
    # Simulate second exercise: Hip Thrust  
    tracker.start_exercise_tracking("hip_thrust", 12, 3)
    
    # Simulate 3 sets of 12 reps each
    for set_num in range(3):
        for rep in range(12):
            form_score = 89 + (rep % 8)  # Varying form scores 89-96%
            tracker.log_rep_completion(form_score)
        tracker.complete_exercise_set()
    
    tracker.finish_exercise_tracking()
    
    # Save the report
    filename = tracker.save_workout_report()
    
    if filename:
        print(f"✅ Test JSON workout report created: {filename}")
        
        # Load and display the report
        tracker.load_workout_report(filename)
        
        # Show the raw JSON structure
        print(f"\n🔍 RAW JSON STRUCTURE:")
        print("=" * 40)
        with open(filename, 'r') as f:
            data = json.load(f)
            # Pretty print first exercise as example
            if data['exercises']:
                print("Example exercise data:")
                print(json.dumps(data['exercises'][0], indent=2))
        
        print(f"\n🤖 AI ANALYSIS READY:")
        print("This JSON file contains structured workout data that AI can analyze for:")
        print("• Progress tracking over time")
        print("• Form improvement recommendations") 
        print("• Workout intensity analysis")
        print("• Muscle group balance assessment")
        print("• Personal training insights")
        print(f"\n📁 Report saved in: workout_reports/")
        print("💡 Use this JSON format to send workout data to AI for analysis!")
        
        return filename
    else:
        print("❌ Failed to create test report")
        return None

if __name__ == "__main__":
    print("🧪 Testing JSON Workout Report System")
    print("=" * 40)
    create_test_json_report()
