"""
Quick TTS test to debug voice issues
"""

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
        
        # Test basic speech
        print("🗣️ Testing basic speech...")
        engine.say("Testing one two three")
        engine.runAndWait()
        
        # Test during loop (similar to workout)
        print("🗣️ Testing in loop...")
        for i in range(3):
            print(f"Loop {i+1}")
            engine.say(f"Number {i+1}")
            engine.runAndWait()
        
        print("✅ TTS test complete")
        
    except Exception as e:
        print(f"❌ TTS test failed: {e}")

if __name__ == "__main__":
    test_tts()
