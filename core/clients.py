"""Centralized API client management with lazy initialization."""

import os
from typing import Optional
from openai import OpenAI
from tavily import TavilyClient


class ClientFactory:
    """Singleton factory for API clients with lazy initialization."""

    _instance = None
    _openai_client: Optional[OpenAI] = None
    _tavily_client: Optional[TavilyClient] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def openai(self) -> OpenAI:
        """Get or create OpenAI client.

        Returns:
            OpenAI client instance

        Raises:
            ValueError: If OPENAI_API_KEY not found in environment
        """
        if self._openai_client is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    @property
    def tavily(self) -> TavilyClient:
        """Get or create Tavily client.

        Returns:
            TavilyClient instance

        Raises:
            ValueError: If TAVILY_API_KEY not found in environment
        """
        if self._tavily_client is None:
            api_key = os.getenv('TAVILY_API_KEY')
            if not api_key:
                raise ValueError("TAVILY_API_KEY not found in environment")
            self._tavily_client = TavilyClient(api_key)
        return self._tavily_client

    def reset(self) -> None:
        """Reset all clients (useful for testing).

        This will force recreation of clients on next access.
        """
        self._openai_client = None
        self._tavily_client = None


# Global instance
clients = ClientFactory()
