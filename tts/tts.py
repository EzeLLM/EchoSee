import sounddevice as sd
import soundfile as sf
from IPython.display import Audio
import numpy as np

def list_audio_devices():
    """Display all available audio output devices."""
    devices = sd.query_devices()
    
    print("Available Audio Devices:")
    print("-----------------------")
    
    for i, device in enumerate(devices):
        # Check if it's an output device
        if device['max_output_channels'] > 0:
            default_mark = " (default)" if device.get('default_output') else ""
            print(f"ID: {i} - {device['name']}{default_mark}")
            print(f"    Channels: {device['max_output_channels']}")
            print(f"    Sample Rate: {device['default_samplerate']}")
            print()
    
    return devices

class TTS:
    """
    Text-to-Speech class using Kokoro for generating audio from text.
    
    Supported language codes:
    - 'a': American English
    - 'b': British English
    - 'e': Spanish
    - 'f': French
    - 'h': Hindi
    - 'i': Italian
    - 'j': Japanese (requires: pip install misaki[ja])
    - 'p': Brazilian Portuguese
    - 'z': Mandarin Chinese (requires: pip install misaki[zh])
    """
    
    def __init__(self, lang_code='a', voice='af_heart'):
        """
        Initialize the TTS engine.
        
        Args:
            lang_code (str): Language code ('a' for American English by default)
            voice (str): Voice ID to use ('af_heart' by default)
        """
        try:
            from kokoro import KPipeline
            self.pipeline = KPipeline(lang_code=lang_code)
            self.initialized = True
        except Exception as e:
            print(f"Warning: Failed to initialize Kokoro TTS engine: {e}")
            print("Some functionality will be limited.")
            self.initialized = False
            
        self.voice = voice
        self.sample_rate = 24000
    
    def generate(self, text, voice=None, speed=1.0, split_pattern=r'\n+'):
        """
        Generate audio from text.
        
        Args:
            text (str): Text to convert to speech
            voice (str, optional): Override the default voice
            speed (float): Speech speed multiplier
            split_pattern (str): Regex pattern to split text
            
        Returns:
            list: List of audio data that can be played directly
        """
        if not self.initialized:
            print("TTS engine not properly initialized. Cannot generate audio.")
            return []
            
        voice = voice or self.voice
        generator = self.pipeline(
            text, 
            voice=voice,
            speed=speed, 
            split_pattern=split_pattern
        )
        
        audio_segments = []
        for _, _, audio in generator:
            audio_segments.append(audio)
            
        return audio_segments
    
    def play(self, text, voice=None, speed=1.0, split_pattern=r'\n+'):
        """
        Generate and play audio from text.
        
        Args:
            text (str): Text to convert to speech
            voice (str, optional): Override the default voice
            speed (float): Speech speed multiplier
            split_pattern (str): Regex pattern to split text
            
        Returns:
            list: List of Audio objects that are automatically played in notebooks
        """
        if not self.initialized:
            print("TTS engine not properly initialized. Cannot play audio.")
            return []
            
        voice = voice or self.voice
        generator = self.pipeline(
            text, 
            voice=voice,
            speed=speed, 
            split_pattern=split_pattern
        )
        
        audio_objects = []
        for i, (_, _, audio) in enumerate(generator):
            audio_obj = Audio(data=audio, rate=self.sample_rate, autoplay=i==0)
            audio_objects.append(audio_obj)
            
        return audio_objects
    
    def save(self, text, filename_prefix="audio", voice=None, speed=1.0, split_pattern=r'\n+'):
        """
        Generate and save audio from text.
        
        Args:
            text (str): Text to convert to speech
            filename_prefix (str): Prefix for saved audio files
            voice (str, optional): Override the default voice
            speed (float): Speech speed multiplier
            split_pattern (str): Regex pattern to split text
            
        Returns:
            list: List of filenames where audio was saved
        """
        if not self.initialized:
            print("TTS engine not properly initialized. Cannot save audio.")
            return []
            
        voice = voice or self.voice
        generator = self.pipeline(
            text, 
            voice=voice,
            speed=speed, 
            split_pattern=split_pattern
        )
        
        filenames = []
        for i, (_, _, audio) in enumerate(generator):
            filename = f"{filename_prefix}_{i}.wav"
            sf.write(filename, audio, self.sample_rate)
            filenames.append(filename)
            
        return filenames

    def play_with_device(self, text, device=None, voice=None, speed=1.0, split_pattern=r'\n+'):
        """
        Generate and play audio through a specific output device.
        
        Args:
            text (str): Text to convert to speech
            device (int or str): Output device ID or name
            voice (str, optional): Override the default voice
            speed (float): Speech speed multiplier
            split_pattern (str): Regex pattern to split text
        """
        if not self.initialized:
            print("TTS engine not properly initialized. Cannot play audio.")
            return
            
        # Get audio data
        audio_segments = self.generate(text, voice, speed, split_pattern)
        
        if not audio_segments:
            return
            
        # Play through specified device (or default if None)
        for audio in audio_segments:
            sd.play(audio, self.sample_rate, device=device)
            sd.wait()  # Wait until audio is finished playing

    def list_devices(self):
        """
        Display a list of all available audio devices.
        
        Returns:
            dict: Dictionary of device information
        """
        return list_audio_devices()


# list_audio_devices()
# ts = TTS(lang_code='a', voice='af_heart')
# ts.play_with_device("""Hey how are you doing""", device=4)