"""Centralized configuration management with singleton pattern."""

import yaml
from typing import Any, Dict, Optional
from pathlib import Path
import threading


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

    def load(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file.

        Args:
            config_path: Optional path to config file. If None, uses default 'config.yml'

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML is invalid
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
