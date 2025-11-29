"""LLM provider abstraction layer with plugin architecture."""

import os
from abc import ABC, abstractmethod
from typing import Optional, Any
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

# Optional imports
try:
    from langchain_deepseek import ChatDeepSeek
    HAS_DEEPSEEK = True
except ImportError:
    HAS_DEEPSEEK = False
    ChatDeepSeek = None

try:
    from smolagents import LiteLLMModel
    HAS_SMOLAGENTS = True
except ImportError:
    HAS_SMOLAGENTS = False
    LiteLLMModel = None

from core.config_manager import config


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def create_chat_model(self, model_name: str) -> BaseChatModel:
        """Create a chat model instance.

        Args:
            model_name: Name of the model to create

        Returns:
            BaseChatModel instance
        """
        pass

    def create_agent_model(self, model_name: str) -> Any:
        """Create an agent model instance (for smolagents).

        Args:
            model_name: Name of the model to create

        Returns:
            LiteLLMModel instance (if smolagents installed)
        """
        raise NotImplementedError("smolagents not installed")


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def create_chat_model(self, model_name: str) -> BaseChatModel:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return ChatOpenAI(api_key=api_key, model=model_name)

    def create_agent_model(self, model_name: str) -> Any:
        if not HAS_SMOLAGENTS:
            raise ImportError("smolagents not installed")
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return LiteLLMModel(f"openai/{model_name}", api_key=api_key)


class DeepSeekProvider(LLMProvider):
    """DeepSeek LLM provider."""

    def create_chat_model(self, model_name: str) -> BaseChatModel:
        if not HAS_DEEPSEEK:
            raise ImportError("langchain_deepseek not installed")
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment")
        return ChatDeepSeek(api_key=api_key, model=model_name)

    def create_agent_model(self, model_name: str) -> Any:
        if not HAS_SMOLAGENTS:
            raise ImportError("smolagents not installed")
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment")
        return LiteLLMModel(f"deepseek/{model_name}", api_key=api_key)


class LLMFactory:
    """Factory for creating LLM instances based on configuration."""

    _providers = {
        'openai': OpenAIProvider(),
        'deepseek': DeepSeekProvider(),
    }

    @classmethod
    def register_provider(cls, name: str, provider: LLMProvider):
        """Register a new LLM provider (for plugins).

        Args:
            name: Provider name
            provider: LLMProvider instance
        """
        cls._providers[name] = provider

    @classmethod
    def create_standard_llm(cls) -> BaseChatModel:
        """Create standard LLM from config.

        Returns:
            BaseChatModel configured from config

        Raises:
            ValueError: If provider is unknown
        """
        llm_config = config.get_section('LLM')
        provider_name = llm_config['provider']
        model_name = llm_config['model']

        provider = cls._providers.get(provider_name)
        if not provider:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

        return provider.create_chat_model(model_name)

    @classmethod
    def create_high_performance_llm(cls) -> BaseChatModel:
        """Create high-performance LLM from config.

        Returns:
            BaseChatModel configured from config

        Raises:
            ValueError: If provider is unknown
        """
        llm_config = config.get_section('LLM')
        provider_name = llm_config['high_performance_provider']
        model_name = llm_config['high_performance_model']

        provider = cls._providers.get(provider_name)
        if not provider:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

        return provider.create_chat_model(model_name)

    @classmethod
    def create_agent_llm(cls) -> Any:
        """Create agent LLM (for smolagents) from config.

        Returns:
            LiteLLMModel configured from config (if smolagents installed)

        Raises:
            ValueError: If provider is unknown
            ImportError: If smolagents not installed
        """
        llm_config = config.get_section('LLM')
        provider_name = llm_config['provider']
        model_name = llm_config['model']

        provider = cls._providers.get(provider_name)
        if not provider:
            raise ValueError(f"Unknown LLM provider: {provider_name}")

        return provider.create_agent_model(model_name)


# Global LLM cache for lazy loading
_llm_cache = {}


def get_llm(llm_type: str = 'standard') -> BaseChatModel:
    """Get LLM instance with caching.

    Args:
        llm_type: 'standard', 'high_performance', or 'agent'

    Returns:
        LLM instance (cached)

    Raises:
        ValueError: If llm_type is unknown
    """
    if llm_type not in _llm_cache:
        if llm_type == 'standard':
            _llm_cache[llm_type] = LLMFactory.create_standard_llm()
        elif llm_type == 'high_performance':
            _llm_cache[llm_type] = LLMFactory.create_high_performance_llm()
        elif llm_type == 'agent':
            _llm_cache[llm_type] = LLMFactory.create_agent_llm()
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")

    return _llm_cache[llm_type]


# Backward compatibility exports - will be lazily loaded
_llm = None
_high_performance_llm = None
_litellm_llm = None


def _ensure_llms_loaded():
    """Ensure all LLMs are loaded (for backward compatibility)."""
    global _llm, _high_performance_llm, _litellm_llm
    if _llm is None:
        _llm = get_llm('standard')
    if _high_performance_llm is None:
        _high_performance_llm = get_llm('high_performance')
    if _litellm_llm is None:
        _litellm_llm = get_llm('agent')


# Properties for lazy loading
@property
def llm():
    """Lazy-loaded standard LLM."""
    global _llm
    if _llm is None:
        _llm = get_llm('standard')
    return _llm


@property
def high_performance_llm():
    """Lazy-loaded high-performance LLM."""
    global _high_performance_llm
    if _high_performance_llm is None:
        _high_performance_llm = get_llm('high_performance')
    return _high_performance_llm


@property
def litellm_llm():
    """Lazy-loaded agent LLM."""
    global _litellm_llm
    if _litellm_llm is None:
        _litellm_llm = get_llm('agent')
    return _litellm_llm
