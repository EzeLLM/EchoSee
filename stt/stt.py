import whisper
import sounddevice as sd
import numpy as np
import CONSTANTS
from utils.utils import open_yaml
import threading
import queue
import readchar  # pip install readchar

class STT():
    def __init__(self):
        self.config = open_yaml(CONSTANTS.CONFIG_PATH, 'STT')
        self.model = whisper.load_model(self.config.get('model_size', 'base.en'))
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.channels = self.config.get('channels', 1)
        self.dtype = 'float32'
        self.recording = False
        self.audio_chunks = []
    
    def transcribe(self, audio):
        """Transcribe audio data using the loaded whisper model"""
        result = self.model.transcribe(audio)
        return result["text"]
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio recording"""
        if self.recording:
            self.audio_chunks.append(indata.copy())
    
    def record_with_key(self, key=' '):
        """Record while a key is held down using a simple polling approach"""
        print(f"Press and hold '{key}' to start recording. Release to stop.")
        print("Press any other key to cancel.")
        
        self.recording = False
        self.audio_chunks = []
        
        # Start audio stream
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self.audio_callback
        )
        
        # Start the stream
        with stream:
            # Wait for key press
            pressed_key = readchar.readkey()
            if pressed_key != key:
                print("Recording canceled")
                return np.array([])
            
            # Start recording
            self.recording = True
            print("Recording started... Release key to stop.")
            
            # Wait for key release
            readchar.readkey()
            self.recording = False
            print("Recording stopped")
        
        # Process the recorded audio
        if not self.audio_chunks:
            print("No audio recorded")
            return np.array([])
        
        # Combine audio chunks
        audio_data = np.concatenate(self.audio_chunks, axis=0)
        
        # Convert to mono if needed
        if self.channels > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        return audio_data.flatten()
    
    def listen_and_transcribe_key(self, key=' '):
        """Record audio while key is pressed and transcribe it"""
        audio_data = self.record_with_key(key)
        if len(audio_data) > 0:
            text = self.transcribe(audio_data)
            return text
        else:
            return "No audio recorded"

# Usage example
if __name__ == "__main__":
    stt = STT()
    print(stt.listen_and_transcribe_key())