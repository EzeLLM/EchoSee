import sounddevice as sd
import soundfile as sf
from IPython.display import Audio
import numpy as np
import io
from pathlib import Path
from openai import OpenAI
import utils.utils as utils
import CONSTANTS
def list_audio_devices():
    """Display all available audio output devices."""
    devices = sd.query_devices()
    
    print("Available Audio Devices:")
    print("-----------------------")
    
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            default_mark = " (default)" if device.get('default_output') else ""
            print(f"ID: {i} - {device['name']}{default_mark}")
            print(f"    Channels: {device['max_output_channels']}")
            print(f"    Sample Rate: {device['default_samplerate']}")
            print()
    
    return devices

class TTS:
    """Text-to-Speech class supporting both Kokoro and OpenAI TTS engines."""
    
    def __init__(self, lang_code=None, voice=None):
        self.config = utils.open_yaml(CONSTANTS.CONFIG_PATH, 'TTS')
        if lang_code is not None:
            self.config['lang_code'] = lang_code
        if voice is not None:
            self.config['voice'] = voice
            
        self.method = self.config.get('method', 'kokoro')
        self.initialized = False
        self.sample_rate = 24000  # Default for Kokoro
        
        if self.method == 'kokoro':
            self._init_kokoro()
        elif self.method == 'openai':
            self._init_openai()
        else:
            raise ValueError(f"Unsupported TTS method: {self.method}")

    def _init_kokoro(self):
        """Initialize Kokoro TTS engine."""
        try:
            from kokoro import KPipeline
            self.lang_code = self.config.get('lang_code', 'a')
            self.voice = self.config.get('voice', 'af_heart')
            self.pipeline = KPipeline(lang_code=self.lang_code)
            self.initialized = True
        except Exception as e:
            print(f"Kokoro initialization failed: {e}")
            self.initialized = False

    def _init_openai(self):
        """Initialize OpenAI TTS engine."""
        try:
            self.client = OpenAI()
            self.voice = self.config.get('voice', 'coral')
            self.model = self.config.get('model', 'gpt-4o-mini-tts')
            self.response_format = self.config.get('response_format', 'mp3')
            self.speed = self.config.get('speed', 1.0)
            
            # Validate response format
            valid_formats = ['pcm', 'mp3', 'opus', 'aac', 'flac', 'wav']
            if self.response_format not in valid_formats:
                raise ValueError(f"Invalid response format. Choose from {valid_formats}")
                
            self.initialized = True
            self.sample_rate = 24000 if self.response_format == 'pcm' else None
        except Exception as e:
            print(f"OpenAI initialization failed: {e}")
            self.initialized = False

    def generate(self, text, voice=None, speed=None, split_pattern=r'\n+'):
        """Generate audio from text using configured TTS engine."""
        if not self.initialized:
            print("TTS engine not initialized")
            return []

        if self.method == 'kokoro':
            return self._generate_kokoro(text, voice, speed, split_pattern)
        return self._generate_openai(text, voice, speed)

    def _generate_kokoro(self, text, voice=None, speed=None, split_pattern=None):
        """Generate audio using Kokoro."""
        voice = voice or self.voice
        generator = self.pipeline(
            text, 
            voice=voice,
            speed=speed or 1.0, 
            split_pattern=split_pattern
        )
        return [audio for _, _, audio in generator]

    def _generate_openai(self, text, voice=None, speed=None):
        """Generate audio using OpenAI."""
        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice or self.voice,
            input=text,
            speed=speed or self.speed,
            response_format=self.response_format,
            instructions="""Voice: Clear, friendly, affectionate
Tone: Neutral and informative, maintaining friendly tone
Delivery: seductive"""
        )
        
        audio_bytes = response.content
        
        # Convert audio bytes to numpy array
        if self.response_format == 'pcm':
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
            self.sample_rate = 24000
        else:
            with io.BytesIO(audio_bytes) as f:
                data, sr = sf.read(f)
                self.sample_rate = int(sr)
                audio_array = data.T if data.ndim > 1 else data
                
        return [audio_array]

    def play(self, text, voice=None, speed=None, split_pattern=r'\n+'):
        """Generate and play audio."""
        audio_segments = self.generate(text, voice, speed, split_pattern)
        return [Audio(data=audio, rate=self.sample_rate) for audio in audio_segments]

    def save(self, text, filename_prefix="audio", voice=None, speed=None, split_pattern=r'\n+'):
        """Generate and save audio to file."""
        if not self.initialized:
            return []
            
        if self.method == 'kokoro':
            return self._save_kokoro(text, filename_prefix, voice, speed, split_pattern)
        return self._save_openai(text, filename_prefix, voice, speed)

    def _save_kokoro(self, text, filename_prefix, voice, speed, split_pattern):
        """Save audio using Kokoro."""
        filenames = []
        generator = self.pipeline(
            text, 
            voice=voice or self.voice,
            speed=speed or 1.0, 
            split_pattern=split_pattern
        )
        for i, (_, _, audio) in enumerate(generator):
            filename = f"{filename_prefix}_{i}.wav"
            sf.write(filename, audio, self.sample_rate)
            filenames.append(filename)
        return filenames

    def _save_openai(self, text, filename_prefix, voice, speed):
        """Save audio using OpenAI."""
        response = self.client.audio.speech.create(
            model=self.model,
            voice=voice or self.voice,
            input=text,
            speed=speed or self.speed,
            response_format=self.response_format,
        )
        
        ext = 'wav' if self.response_format == 'pcm' else self.response_format
        filename = f"{filename_prefix}_0.{ext}"
        
        if self.response_format == 'pcm':
            audio_array = np.frombuffer(response.content, dtype=np.int16)
            sf.write(filename, audio_array, 24000)
        else:
            with open(filename, 'wb') as f:
                f.write(response.content)
                
        return [filename]

    def play_with_device(self, text, device=None, voice=None, speed=None, split_pattern=r'\n+'):
        """Play audio through specific output device."""
        audio_segments = self.generate(text, voice, speed, split_pattern)
        for audio in audio_segments:
            sd.play(audio, self.sample_rate, device=device)
            sd.wait()

    def list_devices(self):
        """List available audio devices."""
        return list_audio_devices()

if __name__ == "__main__":

    
    # Initialize TTS with desired configuration
    tts = TTS()
    tts.play_with_device("tabi, ben seni seviyorum. seni hep seveceğim.")