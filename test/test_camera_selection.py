"""
Quick test for the updated camera selection with dual camera option
"""
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from movenet_tracker import MoveNetWorkoutTracker

def test_camera_selection():
    """Test the updated camera selection"""
    print("📹 Testing Updated Camera Selection")
    print("=" * 50)
    
    tracker = MoveNetWorkoutTracker()
    
    print("You should now see 4 options including dual camera!")
    tracker.select_camera()
    
    print(f"\n✅ Selected Configuration:")
    print(f"   Primary Camera Index: {tracker.primary_camera_index}")
    print(f"   Secondary Camera Index: {tracker.secondary_camera_index}")
    print(f"   Dual Camera Mode: {tracker.dual_camera_mode}")
    
    if tracker.dual_camera_mode:
        print("\n🎬 DUAL CAMERA MODE ACTIVE!")
        print("📹 You can now use both cameras simultaneously")
        print("💡 Perfect for your sky + laptop camera setup")
    else:
        print("\n📱 Single camera mode selected")

if __name__ == "__main__":
    test_camera_selection()
