#!/usr/bin/env python3
"""
Test TTS like it would be used during a workout
"""
import time
import threading

def test_workout_tts():
    """Test TTS exactly like during a workout"""
    try:
        import pyttsx3
        
        print("🔧 Setting up TTS like in workout...")
        tts_engine = pyttsx3.init()
        
        # Configure like workout tracker
        voices = tts_engine.getProperty('voices')
        for voice in voices:
            if 'zira' in voice.name.lower() or 'female' in voice.name.lower():
                tts_engine.setProperty('voice', voice.id)
                print(f"✅ Using: {voice.name}")
                break
        
        tts_engine.setProperty('rate', 150)
        tts_lock = threading.Lock()
        
        def speak_like_workout(text):
            """Speak exactly like the workout tracker"""
            def _speak():
                try:
                    if tts_engine and tts_lock:
                        with tts_lock:
                            tts_engine.say(text)
                            tts_engine.runAndWait()
                except Exception as e:
                    print(f"TTS error: {e}")
            
            print(f"🗣️ {text}")
            thread = threading.Thread(target=_speak)
            thread.daemon = True
            thread.start()
            return thread
        
        # Test startup message
        print("\n🎯 Testing startup...")
        speak_like_workout("Go!")
        time.sleep(3)
        
        # Test rep counting (like during exercise)
        print("\n🎯 Testing rep counting...")
        for rep in range(1, 6):
            print(f"Rep {rep} completed!")
            speak_like_workout(str(rep))
            time.sleep(2)  # Simulate time between reps
        
        # Test completion message
        time.sleep(1)
        speak_like_workout("Workout complete! Great job!")
        
        print("\n✅ Workout TTS test complete")
        print("💡 If you heard all the numbers and messages, TTS is working correctly!")
        
    except Exception as e:
        print(f"❌ TTS test failed: {e}")

if __name__ == "__main__":
    test_workout_tts()
