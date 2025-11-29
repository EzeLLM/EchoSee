"""Custom exceptions for EchoSee application."""


class EchoSeeError(Exception):
    """Base exception for all EchoSee errors."""
    pass


class ConfigurationError(EchoSeeError):
    """Raised when there's a configuration error."""
    pass


class ProviderError(EchoSeeError):
    """Raised when an LLM/Audio provider fails."""
    pass


class AudioError(EchoSeeError):
    """Raised when audio processing fails."""
    pass


class TranscriptionError(AudioError):
    """Raised when speech-to-text transcription fails."""
    pass


class SynthesisError(AudioError):
    """Raised when text-to-speech synthesis fails."""
    pass


class EventError(EchoSeeError):
    """Raised when event scheduling/management fails."""
    pass


class NotificationError(EchoSeeError):
    """Raised when notification delivery fails."""
    pass


class SearchError(EchoSeeError):
    """Raised when web search fails."""
    pass
