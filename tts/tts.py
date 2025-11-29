"""Text-to-Speech service using provider abstraction."""

import re
import numpy as np
import sounddevice as sd
import logging
from typing import Generator, Optional
from core.audio_providers import AudioProviderFactory, OpenAITTSProvider
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
    Includes streaming support for reduced latency.
    """

    # Sentence boundary pattern
    SENTENCE_PATTERN = re.compile(r'([.!?]\s+|\n+)')

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

        # Streaming config
        self.streaming_enabled = tts_config.get('streaming_enabled', True)
        self.min_buffer_chars = tts_config.get('sentence_buffer_chars', 50)

        # Create provider based on config
        self.provider = AudioProviderFactory.create_tts_provider(tts_config)
        self.sample_rate = self.provider.get_sample_rate()

        # Streaming state
        self.is_playing = False
        self.stop_requested = False

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
        device = self._normalize_device(device)
        audio_segments = self.generate(text, **kwargs)
        for audio in audio_segments:
            sd.play(audio, self.sample_rate, device=device)
            sd.wait()

    def _normalize_device(self, device: Optional[int]) -> int:
        """Normalize device to a valid device ID.

        Args:
            device: Device ID or None for default

        Returns:
            Valid device ID
        """
        if device is None:
            try:
                default_device = sd.default.device
                return default_device[1] if isinstance(default_device, tuple) else default_device
            except Exception:
                return -1
        return device

    def _split_sentences(self, text: str) -> list:
        """Split text into sentences, keeping delimiters.

        Args:
            text: Text to split

        Returns:
            List of sentences (may include incomplete final segment)
        """
        # Split while keeping delimiters
        parts = self.SENTENCE_PATTERN.split(text)

        sentences = []
        current = ""

        for i, part in enumerate(parts):
            current += part
            # Check if this part is a delimiter
            if self.SENTENCE_PATTERN.match(part):
                if current.strip():
                    sentences.append(current.strip())
                current = ""

        # Add any remaining text as incomplete sentence
        if current.strip():
            sentences.append(current.strip())

        return sentences

    def supports_streaming(self) -> bool:
        """Check if current provider supports streaming.

        Returns:
            True if streaming is available
        """
        return self.provider.supports_streaming()

    def play_streaming(
        self,
        text_generator: Generator[str, None, None],
        device: Optional[int] = None,
        **kwargs
    ) -> str:
        """Play audio from streaming text generator.

        Buffers text until sentence boundaries, then generates and plays
        audio for each sentence as it completes. This reduces perceived
        latency by starting audio playback before the full response is ready.

        Args:
            text_generator: Generator yielding text chunks
            device: Audio device ID (None for default)
            **kwargs: Provider-specific parameters (voice, speed)

        Returns:
            Full text that was spoken
        """
        if not self.supports_streaming():
            # Fall back to non-streaming mode
            logger.warning("Provider doesn't support streaming, falling back to buffered mode")
            full_text = "".join(text_generator)
            self.play_with_device(full_text, device, **kwargs)
            return full_text

        device = self._normalize_device(device)
        self.is_playing = True
        self.stop_requested = False

        buffer = ""
        full_text = ""

        try:
            for chunk in text_generator:
                if self.stop_requested:
                    logger.info("Streaming playback stopped by request")
                    break

                buffer += chunk
                full_text += chunk

                # Check for sentence boundaries or minimum buffer size
                if self._should_flush_buffer(buffer):
                    sentences = self._split_sentences(buffer)

                    # Play all complete sentences
                    for sentence in sentences[:-1]:
                        if sentence and not self.stop_requested:
                            self._stream_and_play_sentence(sentence, device, **kwargs)

                    # Keep the last (possibly incomplete) part in buffer
                    buffer = sentences[-1] if sentences else ""

            # Play any remaining buffered text
            if buffer.strip() and not self.stop_requested:
                self._stream_and_play_sentence(buffer.strip(), device, **kwargs)

        except Exception as e:
            logger.error(f"Streaming playback error: {e}")
            raise
        finally:
            self.is_playing = False

        return full_text

    def _should_flush_buffer(self, buffer: str) -> bool:
        """Check if buffer should be flushed (has sentence or is too long).

        Args:
            buffer: Current text buffer

        Returns:
            True if buffer should be processed
        """
        # Check for sentence endings
        if any(p in buffer for p in ['.', '!', '?', '\n']):
            return True

        # Force flush if buffer is too large (prevents memory issues)
        if len(buffer) > self.min_buffer_chars * 3:
            return True

        return False

    def _stream_and_play_sentence(
        self,
        sentence: str,
        device: int,
        **kwargs
    ):
        """Stream TTS for a sentence and play it.

        Args:
            sentence: Text to speak
            device: Audio device ID
            **kwargs: Provider-specific parameters
        """
        if not sentence.strip():
            return

        logger.debug(f"Streaming sentence: {sentence[:50]}...")

        try:
            # Use OpenAI streaming
            if isinstance(self.provider, OpenAITTSProvider):
                self._play_openai_streaming(sentence, device, **kwargs)
            else:
                # Fallback for non-streaming providers
                audio_segments = self.provider.generate(sentence, **kwargs)
                for audio in audio_segments:
                    if self.stop_requested:
                        break
                    sd.play(audio, self.sample_rate, device=device)
                    sd.wait()
        except Exception as e:
            logger.error(f"Error streaming sentence: {e}")
            raise

    def _play_openai_streaming(
        self,
        text: str,
        device: int,
        **kwargs
    ):
        """Play OpenAI TTS with efficient streaming.

        Accumulates audio chunks before playing to avoid per-chunk overhead.
        The sd.play()/sd.wait() pattern has significant overhead when called
        many times - combining chunks first is much faster.

        Args:
            text: Text to speak
            device: Audio device ID
            **kwargs: Provider-specific parameters
        """
        try:
            # Collect all audio chunks first - this streams from OpenAI
            # but we accumulate before playback for efficiency
            audio_chunks = []
            for audio_array in self.provider.generate_streaming_array(text, **kwargs):
                if self.stop_requested:
                    break
                audio_chunks.append(audio_array)
            
            if audio_chunks and not self.stop_requested:
                # Combine and play once - much faster than per-chunk playback
                combined_audio = np.concatenate(audio_chunks)
                sd.play(combined_audio, OpenAITTSProvider.PCM_SAMPLE_RATE, device=device)
                sd.wait()
                
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

    def stop_playback(self):
        """Stop current playback (for interruption support)."""
        self.stop_requested = True
        sd.stop()
        logger.info("Playback stopped")

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
