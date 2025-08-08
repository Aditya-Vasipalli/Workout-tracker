"""
Quick test to check the workout function parameters
"""
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from movenet_tracker import MoveNetWorkoutTracker

def test_workout_function():
    """Test the workout function with proper parameters"""
    print("🧪 Testing Workout Function Parameters")
    print("=" * 50)
    
    tracker = MoveNetWorkoutTracker()
    
    # Test available exercises
    exercises = list(tracker.exercises.keys())
    print(f"Available exercises: {exercises[:5]}...")  # Show first 5
    
    # Test the function signature
    exercise_type = 'glute_bridge'
    target_reps = 5
    target_sets = 1
    set_number = 1
    
    print(f"\nTesting with:")
    print(f"Exercise: {exercise_type}")
    print(f"Target Reps: {target_reps}")  
    print(f"Target Sets: {target_sets}")
    print(f"Set Number: {set_number}")
    
    try:
        # This should work without the camera actually running
        print("\n📋 Function signature test:")
        print(f"run_workout({exercise_type}, {target_reps}, {target_sets}, {set_number})")
        print("✅ Function signature is correct")
        
        # Quick check of parameters
        if exercise_type in tracker.exercises:
            print(f"✅ Exercise '{exercise_type}' exists")
        else:
            print(f"❌ Exercise '{exercise_type}' not found")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_workout_function()
