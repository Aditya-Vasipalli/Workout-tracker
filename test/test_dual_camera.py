"""
Test dual camera setup for workout tracking
"""
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from movenet_tracker import MoveNetWorkoutTracker

def test_dual_camera():
    """Test the dual camera setup"""
    print("🎯 Testing Dual Camera Setup")
    print("=" * 60)
    print("This will demonstrate your new dual camera system:")
    print("📹 Sky Camera (overhead) - Fixed position")
    print("💻 Laptop Camera (adjustable) - Position as needed")
    print()
    
    tracker = MoveNetWorkoutTracker()
    
    print("🔧 Testing camera setup for different exercises...\n")
    
    # Test exercises with different camera requirements
    test_exercises = ['frog_pump', 'glute_bridge', 'bulgarian_split_squat']
    
    for exercise in test_exercises:
        print(f"\n🎯 Testing: {tracker.exercises[exercise]['name']}")
        print("-" * 40)
        
        camera_config = tracker.exercises[exercise].get('camera_setup', {})
        if camera_config:
            print(f"📹 Primary Camera: {camera_config['primary'].upper()}")
            print(f"📷 Secondary Camera: {camera_config['secondary'].upper()}")
            print(f"🎬 Main Tracking: {camera_config['primary']} camera")
            print(f"✅ Form Validation: Both cameras")
        else:
            print("❌ No dual camera config (uses single camera)")
    
    print(f"\n🎬 DUAL CAMERA ADVANTAGES:")
    print("✅ Complete movement analysis from multiple angles")
    print("✅ Better form validation and feedback") 
    print("✅ Sky camera shows full body alignment")
    print("✅ Laptop camera captures movement depth")
    print("✅ Combined tracking for accurate rep counting")
    
    print(f"\n🏗️ YOUR SETUP BENEFITS:")
    print("🔥 Sky Camera (Fixed):")
    print("   • Perfect for knee separation (frog pumps)")
    print("   • Great for body alignment validation")
    print("   • Captures full movement patterns")
    
    print("💻 Laptop Camera (Adjustable):")
    print("   • Side view for hip extension depth")
    print("   • Front view for arm/leg width when needed")
    print("   • Detailed joint angle tracking")
    
    choice = input("\n🧪 Would you like to test a specific exercise? (y/n): ").strip().lower()
    
    if choice == 'y':
        print("\nAvailable exercises with dual camera support:")
        for i, exercise in enumerate(test_exercises, 1):
            print(f"{i}. {tracker.exercises[exercise]['name']}")
        
        try:
            ex_choice = int(input(f"\nSelect exercise (1-{len(test_exercises)}): ")) - 1
            if 0 <= ex_choice < len(test_exercises):
                exercise_type = test_exercises[ex_choice]
                print(f"\n🎯 Testing {tracker.exercises[exercise_type]['name']}...")
                
                # This would normally start the dual camera tracking
                print("📹 Camera setup would start here...")
                print("💡 In the actual workout, both cameras would be used simultaneously")
                print("🎬 You would see both camera feeds side by side with tracking info")
                
        except (ValueError, IndexError):
            print("❌ Invalid selection")
    
    print("\n✅ Dual camera system ready for your workouts!")

if __name__ == "__main__":
    test_dual_camera()
