"""Audio provider abstraction for STT and TTS with plugin architecture."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from core.config_manager import config
from core.clients import clients


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""

    @abstractmethod
    def generate(self, text: str, **kwargs) -> List[np.ndarray]:
        """Generate audio from text.

        Args:
            text: Text to convert to speech
            **kwargs: Provider-specific parameters

        Returns:
            List of audio arrays
        """
        pass

    @abstractmethod
    def get_sample_rate(self) -> int:
        """Get audio sample rate.

        Returns:
            Sample rate in Hz
        """
        pass


class OpenAITTSProvider(TTSProvider):
    """OpenAI TTS provider."""

    def __init__(self, tts_config: dict):
        self.client = clients.openai
        self.config = tts_config
        self.voice = tts_config.get('voice', 'shimmer')
        self.model = tts_config.get('model', 'gpt-4o-mini-tts')
        self.response_format = tts_config.get('response_format', 'mp3')
        self.speed = tts_config.get('speed', 1.0)
        self._sample_rate = 24000

    def generate(self, text: str, **kwargs) -> List[np.ndarray]:
        """Generate audio using OpenAI TTS.

        Args:
            text: Text to convert to speech
            **kwargs: Optional overrides (voice, speed, etc.)

        Returns:
            List containing single audio array
        """
        import io
        import soundfile as sf

        response = self.client.audio.speech.create(
            model=self.model,
            voice=kwargs.get('voice', self.voice),
            input=text,
            speed=kwargs.get('speed', self.speed),
            response_format=self.response_format,
        )

        audio_bytes = response.content

        if self.response_format == 'pcm':
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0
        else:
            with io.BytesIO(audio_bytes) as f:
                data, sr = sf.read(f)
                self._sample_rate = int(sr)
                audio_array = data.T if data.ndim > 1 else data

        return [audio_array]

    def get_sample_rate(self) -> int:
        return self._sample_rate


class KokoroTTSProvider(TTSProvider):
    """Kokoro TTS provider (lightweight for Raspberry Pi)."""

    def __init__(self, tts_config: dict):
        from kokoro import KPipeline
        self.config = tts_config
        self.lang_code = tts_config.get('lang_code', 'a')
        self.voice = tts_config.get('voice', 'af_heart')
        self.pipeline = KPipeline(lang_code=self.lang_code)
        self._sample_rate = 24000

    def generate(self, text: str, **kwargs) -> List[np.ndarray]:
        """Generate audio using Kokoro TTS.

        Args:
            text: Text to convert to speech
            **kwargs: Optional overrides (voice, speed, split_pattern)

        Returns:
            List of audio arrays (one per segment)
        """
        voice = kwargs.get('voice', self.voice)
        speed = kwargs.get('speed', 1.0)
        split_pattern = kwargs.get('split_pattern', r'\n+')

        generator = self.pipeline(
            text,
            voice=voice,
            speed=speed,
            split_pattern=split_pattern
        )
        return [audio for _, _, audio in generator]

    def get_sample_rate(self) -> int:
        return self._sample_rate


class STTProvider(ABC):
    """Abstract base class for STT providers."""

    @abstractmethod
    def transcribe(self, audio_file_path: str) -> str:
        """Transcribe audio file to text.

        Args:
            audio_file_path: Path to audio file

        Returns:
            Transcribed text
        """
        pass


class OpenAISTTProvider(STTProvider):
    """OpenAI STT provider (Whisper)."""

    def __init__(self, stt_config: dict):
        self.client = clients.openai
        self.config = stt_config
        self.model = stt_config.get('model', 'whisper-1')
        self.prompt = stt_config.get('prompt', '')

    def transcribe(self, audio_file_path: str) -> str:
        """Transcribe audio using OpenAI Whisper.

        Args:
            audio_file_path: Path to audio file

        Returns:
            Transcribed text
        """
        with open(audio_file_path, 'rb') as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                prompt=self.prompt
            )
        return transcription.text


class AudioProviderFactory:
    """Factory for creating audio providers based on configuration."""

    _tts_providers = {
        'openai': OpenAITTSProvider,
        'kokoro': KokoroTTSProvider,
    }

    _stt_providers = {
        'openai': OpenAISTTProvider,
    }

    @classmethod
    def create_tts_provider(cls, tts_config: Optional[dict] = None) -> TTSProvider:
        """Create TTS provider from config.

        Args:
            tts_config: Optional TTS config dict. If None, loads from ConfigManager

        Returns:
            TTSProvider instance

        Raises:
            ValueError: If provider method is unknown
        """
        if tts_config is None:
            tts_config = config.get_section('TTS')

        method = tts_config.get('method', 'openai')

        provider_class = cls._tts_providers.get(method)
        if not provider_class:
            raise ValueError(f"Unknown TTS provider: {method}")

        return provider_class(tts_config)

    @classmethod
    def create_stt_provider(cls, stt_config: Optional[dict] = None) -> STTProvider:
        """Create STT provider from config.

        Args:
            stt_config: Optional STT config dict. If None, loads from ConfigManager

        Returns:
            STTProvider instance

        Raises:
            ValueError: If provider is unknown
        """
        if stt_config is None:
            stt_config = config.get_section('STT')

        # For now, only OpenAI is supported, but this allows easy extension
        return OpenAISTTProvider(stt_config)

    @classmethod
    def register_tts_provider(cls, name: str, provider_class):
        """Register custom TTS provider.

        Args:
            name: Provider name
            provider_class: TTSProvider class (not instance)
        """
        cls._tts_providers[name] = provider_class

    @classmethod
    def register_stt_provider(cls, name: str, provider_class):
        """Register custom STT provider.

        Args:
            name: Provider name
            provider_class: STTProvider class (not instance)
        """
        cls._stt_providers[name] = provider_class
