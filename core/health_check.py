"""Health check and monitoring.

Provides health checking capabilities for system components,
useful for debugging, monitoring, and deployment verification.
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthChecker:
    """Check health of system components.

    Provides a unified way to check the status of all system
    components including EventManager, API clients, and storage.
    """

    def __init__(self):
        """Initialize health checker."""
        self.start_time = time.time()
        logger.info("HealthChecker initialized")

    def check_all(self) -> Dict[str, Any]:
        """Run all health checks.

        Returns:
            Dictionary with overall status and component statuses
        """
        components = {
            "event_manager": self._check_event_manager(),
            "openai_api": self._check_openai(),
            "tavily_api": self._check_tavily(),
            "config": self._check_config(),
            "conversation_storage": self._check_conversation_storage(),
        }

        # Determine overall status
        all_healthy = all(
            c.get("status") == "healthy"
            for c in components.values()
        )

        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": round(time.time() - self.start_time, 2),
            "components": components
        }

    def _check_event_manager(self) -> Dict[str, Any]:
        """Check EventManager status.

        Returns:
            Component status dictionary
        """
        try:
            from core.app_context import app_context

            if not app_context.is_running:
                return {
                    "status": "stopped",
                    "message": "ApplicationContext not initialized"
                }

            em = app_context.event_manager
            return {
                "status": "healthy" if em.running else "stopped",
                "active_events": len(em.events) if hasattr(em, 'events') else 0
            }
        except RuntimeError as e:
            return {"status": "not_initialized", "message": str(e)}
        except Exception as e:
            logger.error(f"EventManager health check failed: {e}")
            return {"status": "error", "error": str(e)}

    def _check_openai(self) -> Dict[str, Any]:
        """Check OpenAI API connectivity.

        Returns:
            Component status dictionary
        """
        try:
            from core.clients import clients

            client = clients.openai
            if client is None:
                return {"status": "not_configured"}

            # Client exists, assume healthy (actual API call would be expensive)
            return {"status": "healthy", "client_type": type(client).__name__}
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return {"status": "error", "error": str(e)}

    def _check_tavily(self) -> Dict[str, Any]:
        """Check Tavily API connectivity.

        Returns:
            Component status dictionary
        """
        try:
            from core.clients import clients

            client = clients.tavily
            if client is None:
                return {"status": "not_configured"}

            return {"status": "healthy", "client_type": type(client).__name__}
        except Exception as e:
            logger.error(f"Tavily health check failed: {e}")
            return {"status": "error", "error": str(e)}

    def _check_config(self) -> Dict[str, Any]:
        """Check configuration status (file structure only).

        Returns:
            Component status dictionary
        """
        try:
            import yaml
            from pathlib import Path

            # Read config file directly to check structure
            config_path = Path("config.yml")
            if not config_path.exists():
                return {"status": "error", "error": "Config file not found"}

            with open(config_path, 'r') as f:
                all_config = yaml.safe_load(f)

            required_sections = ['LLM', 'TTS', 'STT', 'AgentManager']
            missing = [s for s in required_sections if s not in all_config]

            if missing:
                return {
                    "status": "degraded",
                    "message": f"Missing sections: {missing}"
                }

            return {
                "status": "healthy",
                "sections": list(all_config.keys())
            }
        except FileNotFoundError:
            return {"status": "error", "error": "Config file not found"}
        except Exception as e:
            logger.error(f"Config health check failed: {e}")
            return {"status": "error", "error": str(e)}

    def _check_conversation_storage(self) -> Dict[str, Any]:
        """Check conversation storage status.

        Returns:
            Component status dictionary
        """
        try:
            from core.conversation_storage import ConversationStorage
            from pathlib import Path

            default_path = Path("data/conversations.db")

            if not default_path.exists():
                return {
                    "status": "not_initialized",
                    "message": "Database file does not exist yet"
                }

            storage = ConversationStorage()
            count = storage.get_conversation_count()

            return {
                "status": "healthy",
                "conversation_count": count,
                "db_path": str(storage.db_path)
            }
        except Exception as e:
            logger.error(f"Conversation storage health check failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_quick_status(self) -> str:
        """Get a quick one-line status.

        Returns:
            Simple status string: "healthy", "degraded", or "error"
        """
        try:
            result = self.check_all()
            return result["status"]
        except Exception:
            return "error"


# Global instance
health_checker = HealthChecker()


def check_health() -> Dict[str, Any]:
    """Convenience function to run health check.

    Returns:
        Health check results dictionary
    """
    return health_checker.check_all()


if __name__ == "__main__":
    import json

    print("Running EchoSee Health Check...")
    print("=" * 50)

    result = check_health()

    print(f"\nOverall Status: {result['status'].upper()}")
    print(f"Uptime: {result['uptime_seconds']}s")
    print(f"Timestamp: {result['timestamp']}")

    print("\nComponent Status:")
    print("-" * 50)

    for component, status in result['components'].items():
        status_str = status.get('status', 'unknown')
        icon = "✓" if status_str == "healthy" else "✗" if status_str == "error" else "○"
        print(f"  {icon} {component}: {status_str}")

        # Print additional details for non-healthy components
        if status_str != "healthy":
            for key, value in status.items():
                if key != "status":
                    print(f"      {key}: {value}")

    print("\n" + "=" * 50)
    print("\nFull JSON output:")
    print(json.dumps(result, indent=2))

