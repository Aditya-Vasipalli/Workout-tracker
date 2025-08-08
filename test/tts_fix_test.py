"""
Test the fixed TTS implementation with fresh engines
"""
import pyttsx3
import threading
import time
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from movenet_tracker import MoveNetWorkoutTracker

def test_fixed_tts():
    """Test the new TTS implementation"""
    print("🧪 Testing Fixed TTS Implementation")
    print("=" * 50)
    
    # Create a tracker instance (minimal setup)
    tracker = MoveNetWorkoutTracker()
    
    # Test setup
    tracker.setup_tts()
    
    print(f"TTS Working: {tracker.tts_working}")
    
    if tracker.tts_working:
        print("\n🔊 Testing multiple TTS calls:")
        
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
            time.sleep(2)  # Wait between calls
            
        print("\n✅ TTS test completed!")
        print("Did you hear all the phrases? Check console for any errors.")
        
    else:
        print("❌ TTS setup failed")

def test_direct_engine_creation():
    """Test creating multiple TTS engines directly"""
    print("\n🔧 Testing Direct Engine Creation")
    print("=" * 50)
    
    for i in range(3):
        try:
            print(f"\nCreating engine {i+1}...")
            engine = pyttsx3.init()
            
            # Configure voice
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'zira' in voice.name.lower() or 'female' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            
            engine.setProperty('rate', 150)
            
            # Test phrase
            text = f"Test number {i+1}"
            print(f"Speaking: {text}")
            engine.say(text)
            engine.runAndWait()
            
            # Clean up
            engine.stop()
            del engine
            
            print(f"✅ Engine {i+1} completed successfully")
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Engine {i+1} failed: {e}")

if __name__ == "__main__":
    print("🎯 TTS Fix Verification Test")
    print("=" * 60)
    
    # Test 1: Fixed tracker implementation
    test_fixed_tts()
    
    time.sleep(3)
    
    # Test 2: Direct engine creation
    test_direct_engine_creation()
    
    print("\n🏁 All tests completed!")
    print("If you heard all the phrases, the TTS fix is working!")
