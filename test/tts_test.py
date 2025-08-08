"""
Quick TTS test to debug voice issues
"""
import threading
import time

def test_tts():
    try:
        import pyttsx3
        
        print("🔧 Testing TTS engine...")
        engine = pyttsx3.init()
        
        # List voices
        voices = engine.getProperty('voices')
        print(f"Available voices: {len(voices)}")
        for i, voice in enumerate(voices):
            print(f"  {i}: {voice.name}")
        
        # Set female voice if available
        for voice in voices:
            if 'zira' in voice.name.lower() or 'female' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                print(f"✅ Using voice: {voice.name}")
                break
        
        # Test basic speech
        print("🗣️ Testing basic speech...")
        engine.say("Testing one two three")
        engine.runAndWait()
        
        time.sleep(0.5)  # Small delay
        
        engine.say("Second test")
        engine.runAndWait()
        
        # Test threading approach (like workout tracker)
        print("🗣️ Testing threaded speech...")
        
        def speak_threaded(text):
            def _speak():
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"Thread error: {e}")
            
            print(f"🗣️ {text}")
            thread = threading.Thread(target=_speak)
            thread.daemon = True
            thread.start()
            return thread
        
        # Test multiple threaded calls with delays
        threads = []
        for i in range(5):
            print(f"Loop {i+1}")
            thread = speak_threaded(f"Number {i+1}")
            threads.append(thread)
            time.sleep(2)  # Wait between calls
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        print("✅ TTS test complete")
        
    except Exception as e:
        print(f"❌ TTS test failed: {e}")

if __name__ == "__main__":
    test_tts()
