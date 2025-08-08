#!/usr/bin/env python3
"""
Test the workout reporting functionality
"""

import pickle
from datetime import datetime
import os

def create_test_report():
    """Create a test workout report to demonstrate format"""
    
    # Sample workout session data
    test_session = {
        'session_name': 'Test Glute Workout',
        'session_id': '20250807_143000',
        'start_time': datetime(2025, 8, 7, 14, 30, 0),
        'end_time': datetime(2025, 8, 7, 14, 45, 30),
        'total_duration': 930.0,  # 15.5 minutes
        'exercises': [
            {
                'exercise_name': 'Glute Bridge',
                'exercise_type': 'glute_bridge',
                'target_reps': 15,
                'completed_reps': 15,
                'sets_completed': 3,
                'target_sets': 3,
                'form_scores': [95, 92, 96, 94, 91, 93, 95, 97, 94, 96, 92, 94, 95, 93, 96],
                'avg_form_score': 94.2,
                'start_time': datetime(2025, 8, 7, 14, 30, 0),
                'end_time': datetime(2025, 8, 7, 14, 35, 0),
                'duration': 300.0,
                'primary_muscle': 'Glutes',
                'equipment': 'Bodyweight',
                'difficulty': 'Beginner'
            },
            {
                'exercise_name': 'Hip Thrust',
                'exercise_type': 'hip_thrust',
                'target_reps': 12,
                'completed_reps': 12,
                'sets_completed': 3,
                'target_sets': 3,
                'form_scores': [89, 91, 93, 95, 92, 94, 96, 93, 91, 94, 95, 92],
                'avg_form_score': 92.9,
                'start_time': datetime(2025, 8, 7, 14, 37, 0),
                'end_time': datetime(2025, 8, 7, 14, 42, 0),
                'duration': 300.0,
                'primary_muscle': 'Glutes',
                'equipment': 'Bench + Dumbbell',
                'difficulty': 'Intermediate'
            }
        ],
        'total_exercises': 2,
        'total_reps': 27,
        'total_sets': 6,
        'avg_session_form': 93.6,
        'muscles_worked': ['Glutes']
    }
    
    # Create reports directory
    if not os.path.exists('workout_reports'):
        os.makedirs('workout_reports')
    
    # Save test report
    filename = os.path.join('workout_reports', 'test_workout_20250807_143000.bin')
    with open(filename, 'wb') as f:
        pickle.dump(test_session, f)
    
    print(f"✅ Test workout report created: {filename}")
    return filename

def read_test_report(filename):
    """Read and display the test report"""
    try:
        with open(filename, 'rb') as f:
            session_data = pickle.load(f)
        
        print(f"\n📊 WORKOUT REPORT: {session_data['session_name']}")
        print("=" * 60)
        print(f"📅 Date: {session_data['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
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
        
        print("\n🤖 AI ANALYSIS READY:")
        print("This binary file contains structured workout data that AI can analyze for:")
        print("• Progress tracking over time")
        print("• Form improvement recommendations") 
        print("• Workout intensity analysis")
        print("• Muscle group balance assessment")
        print("• Personal training insights")
        
    except Exception as e:
        print(f"❌ Error reading report: {e}")

if __name__ == "__main__":
    print("🧪 Testing Workout Report System")
    print("=" * 40)
    
    # Create test report
    test_file = create_test_report()
    
    # Read and display
    read_test_report(test_file)
    
    print(f"\n📁 Report saved in: workout_reports/")
    print("💡 Use this format to send workout data to AI for analysis!")
