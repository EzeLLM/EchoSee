"""Application context for managing global state and lifecycle."""

import atexit
from typing import Optional
from event_manager.event_manager import EventManager


class ApplicationContext:
    """Manages application lifecycle and shared resources.

    This class replaces module-level side effects with explicit lifecycle
    management. Instead of auto-starting services on import, they are
    started explicitly when the application context is initialized.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._event_manager: Optional[EventManager] = None
        self._is_running = False
        self._initialized = True

    def initialize(self):
        """Initialize application context and start services.

        This should be called once at application startup, before using
        any services like the EventManager.
        """
        if self._is_running:
            return

        # Start EventManager
        self._event_manager = EventManager()
        self._event_manager.start()

        # Register cleanup handler
        atexit.register(self.shutdown)

        self._is_running = True

    def shutdown(self):
        """Shutdown application context and cleanup resources.

        This is automatically called via atexit when the application exits.
        """
        if not self._is_running:
            return

        print("Exiting...")
        if self._event_manager:
            self._event_manager.stop()
            print("Event manager stopped.")

        self._is_running = False

    @property
    def event_manager(self) -> EventManager:
        """Get EventManager instance.

        Returns:
            EventManager instance

        Raises:
            RuntimeError: If ApplicationContext not initialized
        """
        if not self._is_running:
            raise RuntimeError(
                "ApplicationContext not initialized. "
                "Call app_context.initialize() first."
            )
        return self._event_manager

    @property
    def is_running(self) -> bool:
        """Check if application context is running.

        Returns:
            True if initialized and running
        """
        return self._is_running


# Global instance
app_context = ApplicationContext()
