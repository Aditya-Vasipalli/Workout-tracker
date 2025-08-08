"""
Test the new extended camera display layout
"""
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from movenet_tracker import MoveNetWorkoutTracker

def test_extended_display():
    """Test the new extended camera display with text area"""
    print("🖥️ Testing Extended Display Layout")
    print("=" * 60)
    print("This will test the new camera layout with:")
    print("📹 Camera feed on the left")
    print("🖤 Black text area on the right with workout info")
    print("\nStarting camera test...")
    
    tracker = MoveNetWorkoutTracker()
    
    # First select camera
    tracker.select_camera()
    
    # Test the camera with new layout
    print("\n🎬 Starting camera test with extended display...")
    print("You should see:")
    print("  • Camera feed on the left side")
    print("  • Black area on the right with text information")
    print("  • All text clearly visible against black background")
    print("\nPress 'Q' to quit the test")
    
    tracker.test_camera_view()
    
    print("✅ Extended display test completed!")

if __name__ == "__main__":
    test_extended_display()
