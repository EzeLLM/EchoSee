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


# Note: Module-level @property decorators don't work - they only work on class methods.
# The following functions are kept for reference but actual exports use direct assignment below.


def get_event_manager():
    """Get EventManager instance.

    DEPRECATED: Use core.app_context.app_context.event_manager instead.

    Returns:
        EventManager instance

    Raises:
        RuntimeError: If app_context not initialized
    """
    return app_context.event_manager


# Direct LLM initialization (env is already validated above)
# These are actual LLM instances that work with LangChain
llm = get_llm('standard')
high_performance_llm = get_llm('high_performance')

# Agent LLM (for smolagents) - lazy load to avoid import issues
_litellm_llm = None

def get_litellm():
    """Get LiteLLM instance for smolagents."""
    global _litellm_llm
    if _litellm_llm is None:
        _litellm_llm = get_llm('agent')
    return _litellm_llm

# Backward-compatible export - alias to the getter function
# Code that imports litellm_llm should call it as litellm_llm() to get the instance
litellm_llm = get_litellm

# Backward-compatible config exports
# These are functions that return the config sections
def tts_config():
    """Get TTS configuration section.
    
    DEPRECATED: Use core.config_manager.config.get_section('TTS') instead.
    """
    return _config_manager.get_section('TTS')


def llm_config():
    """Get LLM configuration section.
    
    DEPRECATED: Use core.config_manager.config.get_section('LLM') instead.
    """
    return _config_manager.get_section('LLM')


# Event manager instance (lazy-loaded via app_context)
class _LazyEventManager:
    """Wrapper for lazy-loading EventManager via app_context."""
    def __getattr__(self, name):
        return getattr(app_context.event_manager, name)


event_manager_instance = _LazyEventManager()


# Note: Application context must be initialized explicitly in main()
# Call: app_context.initialize() at the start of your application
