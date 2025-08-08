"""
Test the improved TTS implementation with PowerShell
"""
import time
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from movenet_tracker import MoveNetWorkoutTracker

def test_improved_tts():
    """Test the improved TTS implementation with PowerShell"""
    print("🎯 Testing Improved TTS with PowerShell")
    print("=" * 60)
    
    # Create a tracker instance
    tracker = MoveNetWorkoutTracker()
    
    # Test setup
    tracker.setup_tts()
    
    print(f"TTS Working: {tracker.tts_working}")
    if hasattr(tracker, 'tts_method'):
        print(f"TTS Method: {tracker.tts_method}")
    
    if tracker.tts_working:
        print("\n🔊 Testing workout flow TTS:")
        
        # Test sequence - simulating workout flow
        test_phrases = [
            "Get ready for squats!",
            "Go!",
            "1",
            "2", 
            "3",
            "4",
            "5",
            "Great job!",
            "Exercise complete!"
        ]
        
        for i, phrase in enumerate(test_phrases):
            print(f"\nTest {i+1}: '{phrase}'")
            tracker.speak(phrase)
            time.sleep(2.5)  # Wait between calls
            
        print("\n✅ Improved TTS test completed!")
        print("🎧 Did you hear all the phrases clearly?")
        
    else:
        print("❌ TTS setup failed")

if __name__ == "__main__":
    test_improved_tts()
