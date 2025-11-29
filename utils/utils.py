# utils/utils.py
"""
Utility functions and backward-compatible exports.

This module has been refactored to remove module-level side effects.
LLMs and config are now managed by core modules (core/llm_factory.py, core/config_manager.py).
"""

import yaml
import dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

# Validate required environment variables before proceeding
from utils.env_validation import require_env_vars
require_env_vars()

# Import core modules
from core.config_manager import config as _config_manager
from core.llm_factory import get_llm
from core.app_context import app_context

# Constants (kept for backward compatibility)
CONFIG_PATH = 'config.yml'


def open_yaml(file_path, key=None):
    """Safe YAML file loader with optional key filtering.

    DEPRECATED: Use core.config_manager.config instead.
    This function is kept for backward compatibility only.

    Args:
        file_path: Path to YAML file
        key: Optional key to filter

    Returns:
        Loaded YAML data or None on error
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data.get(key) if key else data
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return None
    except AttributeError:
        logger.error(f"Key {key} not found in file {file_path}.")
        return None


# Backward-compatible exports for LLMs (lazy-loaded via LLMFactory)
@property
def llm():
    """Standard LLM instance (lazy-loaded).

    Returns:
        BaseChatModel configured from config
    """
    return get_llm('standard')


@property
def high_performance_llm():
    """High-performance LLM instance (lazy-loaded).

    Returns:
        BaseChatModel configured from config
    """
    return get_llm('high_performance')


@property
def litellm_llm():
    """Agent LLM instance for smolagents (lazy-loaded).

    Returns:
        LiteLLMModel configured from config
    """
    return get_llm('agent')


# Backward-compatible config exports (use ConfigManager)
@property
def tts_config():
    """TTS configuration section.

    DEPRECATED: Use core.config_manager.config.get_section('TTS') instead.

    Returns:
        TTS configuration dict
    """
    return _config_manager.get_section('TTS')


@property
def llm_config():
    """LLM configuration section.

    DEPRECATED: Use core.config_manager.config.get_section('LLM') instead.

    Returns:
        LLM configuration dict
    """
    return _config_manager.get_section('LLM')


def get_event_manager():
    """Get EventManager instance.

    DEPRECATED: Use core.app_context.app_context.event_manager instead.

    Returns:
        EventManager instance

    Raises:
        RuntimeError: If app_context not initialized
    """
    return app_context.event_manager


# Module-level exports (for backward compatibility)
# These are NOT auto-initialized; they're created on first access
_llm = None
_high_performance_llm = None
_litellm_llm = None
_event_manager_instance = None


# For files that import directly: from utils.utils import llm
# We need to provide actual values, not properties
# So let's create wrapper that lazy-loads on first access
class _LazyLLM:
    """Wrapper for lazy-loading LLMs."""
    def __init__(self, llm_type):
        self._llm_type = llm_type
        self._instance = None

    def __getattr__(self, name):
        if self._instance is None:
            self._instance = get_llm(self._llm_type)
        return getattr(self._instance, name)

    def __call__(self, *args, **kwargs):
        if self._instance is None:
            self._instance = get_llm(self._llm_type)
        return self._instance(*args, **kwargs)


# Create lazy-loaded instances that can be imported directly
llm = _LazyLLM('standard')
high_performance_llm = _LazyLLM('high_performance')
litellm_llm = _LazyLLM('agent')

# Event manager instance (lazy-loaded via app_context)
class _LazyEventManager:
    """Wrapper for lazy-loading EventManager via app_context."""
    def __getattr__(self, name):
        return getattr(app_context.event_manager, name)


event_manager_instance = _LazyEventManager()


# Note: Application context must be initialized explicitly in main()
# Call: app_context.initialize() at the start of your application
