"""
Alternative TTS implementation using Windows built-in speech
"""
import subprocess
import threading
import time

class WindowsTTS:
    """Windows-specific TTS using PowerShell's speech synthesizer"""
    
    def __init__(self):
        self.tts_lock = threading.Lock()
    
    def speak(self, text):
        """Speak text using Windows PowerShell speech synthesizer"""
        def tts_worker():
            try:
                # Use PowerShell's built-in speech synthesizer
                ps_command = f'''
                Add-Type -AssemblyName System.Speech;
                $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer;
                $synth.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female);
                $synth.Rate = 0;
                $synth.Speak("{text}");
                '''
                
                subprocess.run([
                    "powershell", "-Command", ps_command
                ], capture_output=True, text=True, timeout=10)
                
            except Exception as e:
                print(f"⚠️ PowerShell TTS error: {e}")
                # Fallback to beep
                try:
                    import winsound
                    winsound.Beep(1000, 200)
                except:
                    pass
        
        with self.tts_lock:
            print(f"🔊 Speaking: {text}")
            thread = threading.Thread(target=tts_worker)
            thread.daemon = True
            thread.start()
            # Wait briefly to avoid overlap
            time.sleep(0.1)

def test_windows_tts():
    """Test the Windows PowerShell TTS"""
    print("🧪 Testing Windows PowerShell TTS")
    print("=" * 50)
    
    tts = WindowsTTS()
    
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
        tts.speak(phrase)
        time.sleep(3)  # Wait between calls
    
    print("\n✅ Windows TTS test completed!")

if __name__ == "__main__":
    test_windows_tts()
