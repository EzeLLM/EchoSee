"""Centralized configuration management with singleton pattern."""

import os
import yaml
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validate configuration structure and values."""

    @staticmethod
    def validate(config: Dict[str, Any]) -> List[str]:
        """Validate config and return list of errors.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check required sections
        required_sections = ['LLM', 'TTS', 'STT', 'AgentManager']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")

        # Validate LLM config
        if 'LLM' in config:
            llm = config['LLM']
            if 'provider' not in llm:
                errors.append("LLM.provider is required")
            elif llm['provider'] not in ['openai', 'deepseek']:
                errors.append(f"Invalid LLM.provider: {llm['provider']}")

            if 'model' not in llm:
                errors.append("LLM.model is required")

            # Check API keys for providers
            if llm.get('provider') == 'openai' and not os.getenv('OPENAI_API_KEY'):
                errors.append("OPENAI_API_KEY required for OpenAI provider")
            if llm.get('provider') == 'deepseek' and not os.getenv('DEEPSEEK_API_KEY'):
                errors.append("DEEPSEEK_API_KEY required for DeepSeek provider")

        # Validate TTS config
        if 'TTS' in config:
            tts = config['TTS']
            if 'method' not in tts:
                errors.append("TTS.method is required")
            elif tts['method'] not in ['openai', 'kokoro']:
                errors.append(f"Invalid TTS.method: {tts['method']}")

        # Validate STT config
        if 'STT' in config:
            stt = config['STT']
            if 'sample_rate' in stt and not isinstance(stt['sample_rate'], int):
                errors.append("STT.sample_rate must be an integer")

        # Validate AgentManager config
        if 'AgentManager' in config:
            am = config['AgentManager']
            if 'clear_time' in am:
                if not isinstance(am['clear_time'], (int, float)):
                    errors.append("AgentManager.clear_time must be a number")
                elif am['clear_time'] <= 0:
                    errors.append("AgentManager.clear_time must be positive")

        return errors


class ConfigManager:
    """Singleton configuration manager with lazy loading and caching."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._config: Optional[Dict[str, Any]] = None
        self._config_path = Path("config.yml")
        self._initialized = True

    def load(self, config_path: Optional[str] = None, validate: bool = True) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Optional path to config file. If None, uses default 'config.yml'
            validate: Whether to validate config after loading (default True)

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML is invalid or validation fails
        """
        if config_path:
            self._config_path = Path(config_path)

        try:
            with open(self._config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self._config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

        # Validate after loading
        if validate:
            errors = ConfigValidator.validate(self._config)
            if errors:
                error_msg = "Configuration validation failed:\n" + "\n".join(
                    f"  - {e}" for e in errors
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys like 'LLM.model').

        Args:
            key: Configuration key, supports dot notation for nested values
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if self._config is None:
            self.load()

        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.

        Args:
            section: Section name (e.g., 'LLM', 'TTS', 'STT')

        Returns:
            Dictionary with section configuration

        Raises:
            KeyError: If section doesn't exist
        """
        result = self.get(section)
        if result is None:
            raise KeyError(f"Configuration section '{section}' not found")
        return result

    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = None
        self.load()

    @property
    def all(self) -> Dict[str, Any]:
        """Get entire configuration.

        Returns:
            Copy of entire configuration dictionary
        """
        if self._config is None:
            self.load()
        return self._config.copy()


# Global instance
config = ConfigManager()


# Convenience functions for backward compatibility
def get_config(key: str = None, default: Any = None) -> Any:
    """Get configuration value. If key is None, returns entire config.

    Args:
        key: Optional configuration key (supports dot notation)
        default: Default value if key not found

    Returns:
        Configuration value, section, or entire config
    """
    if key is None:
        return config.all
    return config.get(key, default)


def get_config_section(section: str) -> Dict[str, Any]:
    """Get configuration section.

    Args:
        section: Section name

    Returns:
        Dictionary with section configuration
    """
    return config.get_section(section)
