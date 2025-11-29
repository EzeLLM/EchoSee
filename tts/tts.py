"""Text-to-Speech service using provider abstraction."""

import sounddevice as sd
import logging
from core.audio_providers import AudioProviderFactory
from core.config_manager import config

logger = logging.getLogger(__name__)


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
    """Text-to-Speech service using provider abstraction.

    Supports multiple TTS providers (OpenAI, Kokoro, etc.) via AudioProviderFactory.
    The provider is selected based on the 'method' setting in config.yml.
    """

    def __init__(self, lang_code=None, voice=None):
        """Initialize TTS with optional overrides.

        Args:
            lang_code: Optional language code override (for Kokoro)
            voice: Optional voice override
        """
        # Get TTS config
        tts_config = config.get_section('TTS').copy()

        # Apply overrides if provided
        if lang_code is not None:
            tts_config['lang_code'] = lang_code
        if voice is not None:
            tts_config['voice'] = voice

        # Create provider based on config
        self.provider = AudioProviderFactory.create_tts_provider(tts_config)
        self.sample_rate = self.provider.get_sample_rate()

    def generate(self, text: str, **kwargs):
        """Generate audio from text.

        Args:
            text: Text to convert to speech
            **kwargs: Provider-specific parameters (voice, speed, split_pattern, etc.)

        Returns:
            List of audio arrays
        """
        return self.provider.generate(text, **kwargs)

    def play_with_device(self, text: str, device=None, **kwargs):
        """Generate and play audio through specific device.

        Args:
            text: Text to convert to speech
            device: Audio device ID (None for default)
            **kwargs: Provider-specific parameters
        """
        if device is None:
            try:
                default_device = sd.default.device
                device = default_device[1] if isinstance(default_device, tuple) else default_device
            except Exception:
                device = -1

        audio_segments = self.generate(text, **kwargs)
        for audio in audio_segments:
            sd.play(audio, self.sample_rate, device=device)
            sd.wait()

    @staticmethod
    def list_devices():
        """List available audio devices.

        Returns:
            List of audio device info
        """
        return list_audio_devices()


if __name__ == '__main__':
    # Test TTS
    tts = TTS()
    print(f"Initialized TTS with sample rate: {tts.sample_rate}")

    # List devices
    print("\nAvailable devices:")
    TTS.list_devices()

    # Test generation (but don't play to avoid noise in tests)
    test_text = "Hello, this is a test."
    audio_segments = tts.generate(test_text)
    print(f"\nGenerated {len(audio_segments)} audio segment(s) for: '{test_text}'")
