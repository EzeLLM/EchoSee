"""Speech-to-Text service using provider abstraction."""

import tempfile
import logging
from core.config_manager import config
from core.audio_providers import AudioProviderFactory
import threading
import pyaudio
import readchar
import os
import subprocess

logger = logging.getLogger(__name__)


class STT:
    """Speech-to-Text service using provider abstraction.

    Handles audio recording and transcription. The transcription provider
    is selected based on configuration (currently OpenAI Whisper).
    """

    def __init__(self):
        """Initialize STT service."""
        stt_config = config.get_section('STT')

        # Create STT provider
        self.provider = AudioProviderFactory.create_stt_provider(stt_config)

        # Audio recording settings
        self.sample_rate = stt_config.get('sample_rate', 16000)
        self.channels = stt_config.get('channels', 1)
        self.format = pyaudio.paInt16
        self.chunk = 1024
        self.recording = False

        # Initialize PyAudio in try/except to handle potential initialization errors
        try:
            self.p = pyaudio.PyAudio()
        except Exception as e:
            logger.warning(f"Could not initialize PyAudio: {e}")
            self.p = None

    def transcribe(self, audio_file_path: str) -> str:
        """Transcribe audio file using configured provider.

        Args:
            audio_file_path: Path to audio file

        Returns:
            Transcribed text
        """
        return self.provider.transcribe(audio_file_path)
    
    def record_with_key(self, key=' '):
        """Record while a key is held down and encode directly to MP3"""
        # Check if PyAudio was initialized successfully
        if self.p is None:
            print("Error: PyAudio is not initialized")
            return None
            
        print(f"Press and hold '{key}' to start recording. Release to stop.")
        print("Press any other key to cancel.")
        
        # Create a temporary file for the MP3
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Wait for key press
        pressed_key = readchar.readkey()
        if pressed_key != key:
            print("Recording canceled")
            os.unlink(temp_file_path)
            return None
        
        # Set up FFmpeg process for direct MP3 encoding
        ffmpeg_cmd = [
            'ffmpeg',
            '-f', 's16le',            # Input format (16-bit PCM)
            '-ar', str(self.sample_rate),  # Sample rate
            '-ac', str(self.channels),     # Channels
            '-i', 'pipe:0',           # Read from stdin
            '-c:a', 'libmp3lame',     # MP3 codec
            '-b:a', '128k',           # Bitrate
            '-y',                     # Overwrite output
            temp_file_path            # Output file
        ]
        
        try:
            ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd, 
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Start recording
            stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            self.recording = True
            print("Recording started... Release key to stop.")
            
            # Record in a separate thread
            def record_thread():
                while self.recording:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    ffmpeg_process.stdin.write(data)
            
            thread = threading.Thread(target=record_thread)
            thread.start()
            
            # Wait for key release
            readchar.readkey()
            self.recording = False
            thread.join()
            
            print("Recording stopped")
            
            # Clean up
            stream.stop_stream()
            stream.close()
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            
            # Check if the file exists and has content
            if os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 0:
                return temp_file_path
            else:
                print("No audio recorded or encoding failed")
                os.unlink(temp_file_path)
                return None
                
        except Exception as e:
            print(f"Error during recording: {e}")
            if 'ffmpeg_process' in locals():
                try:
                    ffmpeg_process.terminate()
                except:
                    pass
            os.unlink(temp_file_path)
            return None
    
    def listen_and_transcribe_key(self, key=' '):
        """Record audio while key is pressed and transcribe it"""
        audio_file_path = self.record_with_key(key)
        if audio_file_path:
            try:
                text = self.transcribe(audio_file_path)
                return text
            finally:
                # Clean up the temporary file
                os.unlink(audio_file_path)
        else:
            return "No audio recorded"
    
    def __del__(self):
        """Clean up PyAudio when the object is destroyed"""
        try:
            if hasattr(self, 'p') and self.p is not None:
                self.p.terminate()
        except Exception as e:
            print(f"Warning: Error while terminating PyAudio: {e}")

# Usage example
if __name__ == "__main__":
    stt = STT()
    print(stt.listen_and_transcribe_key())