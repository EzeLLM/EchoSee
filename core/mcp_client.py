"""MCP (Model Context Protocol) client management for loading external tools.

This module provides a singleton MCP client manager that:
- Loads MCP server configurations from config.yml
- Connects to MCP servers using various transports (stdio, http, sse)
- Loads tools from connected servers for use with LangChain agents
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from core.config_manager import config

logger = logging.getLogger(__name__)

# Check if MCP adapters are installed
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    MultiServerMCPClient = None
    load_mcp_tools = None


class MCPClientManager:
    """Singleton manager for MCP server connections and tool loading.
    
    This class manages connections to MCP servers defined in config.yml
    and provides tools that can be used with LangChain agents.
    
    As of langchain-mcp-adapters 0.1.0, the client is NOT used as a context
    manager. Instead, we directly call get_tools() on the client.
    
    Usage:
        # Sync context (blocking):
        tools = mcp_manager.get_tools_sync()
        
        # Async context:
        tools = await mcp_manager.get_tools()
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
            
        self._client: Optional[MultiServerMCPClient] = None
        self._tools: List[Any] = []
        self._server_configs: Dict[str, Dict] = {}
        self._is_connected = False
        self._initialized = True
        self._load_config()
    
    def _load_config(self) -> None:
        """Load MCP configuration from config.yml."""
        try:
            mcp_config = config.get('MCP', {})
            self._enabled = mcp_config.get('enabled', True)
            self._server_configs = mcp_config.get('servers', {}) or {}
            
            # Filter out None or empty configs
            self._server_configs = {
                k: v for k, v in self._server_configs.items() 
                if v is not None and isinstance(v, dict)
            }
            
            logger.debug(f"Loaded MCP config: enabled={self._enabled}, servers={list(self._server_configs.keys())}")
        except Exception as e:
            logger.warning(f"Could not load MCP config: {e}")
            self._enabled = False
            self._server_configs = {}
    
    @property
    def enabled(self) -> bool:
        """Check if MCP is enabled in config."""
        return self._enabled and HAS_MCP
    
    @property
    def has_servers(self) -> bool:
        """Check if any MCP servers are configured."""
        return bool(self._server_configs)
    
    @property
    def server_names(self) -> List[str]:
        """Get list of configured server names."""
        return list(self._server_configs.keys())
    
    @property
    def is_connected(self) -> bool:
        """Check if currently connected to MCP servers."""
        return self._is_connected
    
    def _build_client_config(self) -> Dict[str, Dict]:
        """Build MultiServerMCPClient configuration from config.yml format.
        
        Converts config.yml server definitions to the format expected by
        langchain-mcp-adapters.
        """
        client_config = {}
        
        for name, server_conf in self._server_configs.items():
            transport = server_conf.get('transport', 'stdio')
            
            if transport == 'stdio':
                # Stdio transport configuration
                client_config[name] = {
                    'transport': 'stdio',
                    'command': server_conf.get('command', 'python'),
                    'args': server_conf.get('args', []),
                }
                # Add environment variables if specified
                if 'env' in server_conf:
                    client_config[name]['env'] = server_conf['env']
                    
            elif transport == 'streamable_http':
                # HTTP transport configuration
                client_config[name] = {
                    'transport': 'streamable_http',
                    'url': server_conf.get('url'),
                }
                
            elif transport == 'sse':
                # Server-Sent Events transport
                client_config[name] = {
                    'transport': 'sse',
                    'url': server_conf.get('url'),
                }
            else:
                logger.warning(f"Unknown MCP transport '{transport}' for server '{name}'")
                continue
        
        return client_config
    
    async def connect(self) -> None:
        """Connect to all configured MCP servers.
        
        This method initializes connections to all MCP servers defined in config.
        As of langchain-mcp-adapters 0.1.0, we create the client and call get_tools().
        """
        if not self.enabled:
            logger.info("MCP is disabled or langchain-mcp-adapters not installed")
            return
        
        if not self.has_servers:
            logger.debug("No MCP servers configured")
            return
        
        if self._is_connected:
            logger.debug("Already connected to MCP servers")
            return
        
        try:
            client_config = self._build_client_config()
            if not client_config:
                logger.warning("No valid MCP server configurations found")
                return
            
            logger.info(f"Connecting to MCP servers: {list(client_config.keys())}")
            self._client = MultiServerMCPClient(client_config)
            
            # As of 0.1.0, get_tools() is async and handles connection internally
            self._tools = await self._client.get_tools()
            self._is_connected = True
            
            logger.info(f"Loaded {len(self._tools)} tools from MCP servers")
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP servers: {e}")
            self._is_connected = False
            self._tools = []
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from all MCP servers."""
        # As of 0.1.0, the client manages connections internally
        self._is_connected = False
        self._tools = []
        self._client = None
    
    async def get_tools(self) -> List[Any]:
        """Get tools from connected MCP servers.
        
        Returns:
            List of LangChain-compatible tools from MCP servers
        """
        if not self._is_connected:
            await self.connect()
        return self._tools
    
    def get_tools_sync(self) -> List[Any]:
        """Synchronously get tools from MCP servers.
        
        This method handles the async connection in a sync context,
        suitable for use in non-async code like AgentManager.__init__.
        
        Handles both cases:
        - When no event loop is running: creates a new one
        - When an event loop is already running: uses nest_asyncio if available,
          otherwise falls back to creating a separate task
        
        Returns:
            List of LangChain-compatible tools from MCP servers
        """
        if not self.enabled:
            logger.debug("MCP disabled, returning empty tools list")
            return []
        
        if not self.has_servers:
            logger.debug("No MCP servers configured")
            return []
        
        if self._is_connected:
            return self._tools
        
        try:
            # Check if there's already a running event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - try to use nest_asyncio
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    loop.run_until_complete(self.connect())
                    return self._tools
                except ImportError:
                    # nest_asyncio not available - log warning and skip MCP
                    logger.warning(
                        "MCP tools cannot be loaded in async context without nest_asyncio. "
                        "Install with: pip install nest_asyncio"
                    )
                    return []
            except RuntimeError:
                # No event loop running - create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.connect())
                    return self._tools
                finally:
                    # Don't close the loop - we need it for tool invocations
                    pass
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")
            return []
    
    def get_server_info(self) -> Dict[str, Dict]:
        """Get information about configured servers.
        
        Returns:
            Dictionary with server names and their configurations
        """
        info = {}
        for name, conf in self._server_configs.items():
            info[name] = {
                'transport': conf.get('transport', 'stdio'),
                'connected': self._is_connected,
            }
            if conf.get('transport') == 'stdio':
                info[name]['command'] = conf.get('command', 'python')
            else:
                info[name]['url'] = conf.get('url', '')
        return info
    
    async def test_server(self, server_name: str) -> Dict[str, Any]:
        """Test connection to a specific MCP server.
        
        Args:
            server_name: Name of the server to test
            
        Returns:
            Dictionary with test results including status, tools, and any errors
        """
        if server_name not in self._server_configs:
            return {
                'status': 'error',
                'error': f"Server '{server_name}' not found in configuration"
            }
        
        server_conf = self._server_configs[server_name]
        client_config = {}
        
        # Build single-server config
        transport = server_conf.get('transport', 'stdio')
        if transport == 'stdio':
            client_config[server_name] = {
                'transport': 'stdio',
                'command': server_conf.get('command', 'python'),
                'args': server_conf.get('args', []),
            }
            if 'env' in server_conf:
                client_config[server_name]['env'] = server_conf['env']
        elif transport in ('streamable_http', 'sse'):
            client_config[server_name] = {
                'transport': transport,
                'url': server_conf.get('url'),
            }
        
        try:
            # As of 0.1.0, don't use context manager
            client = MultiServerMCPClient(client_config)
            tools = await client.get_tools()
            return {
                'status': 'connected',
                'server_name': server_name,
                'transport': transport,
                'tools': [
                    {
                        'name': t.name,
                        'description': t.description[:100] + '...' if len(t.description) > 100 else t.description
                    }
                    for t in tools
                ],
                'tool_count': len(tools)
            }
        except Exception as e:
            return {
                'status': 'error',
                'server_name': server_name,
                'transport': transport,
                'error': str(e)
            }
    
    async def test_all_servers(self) -> List[Dict[str, Any]]:
        """Test connections to all configured MCP servers.
        
        Returns:
            List of test results for each server
        """
        results = []
        for server_name in self._server_configs:
            result = await self.test_server(server_name)
            results.append(result)
        return results
    
    def reload_config(self) -> None:
        """Reload MCP configuration from config.yml.
        
        Note: This will require reconnection to apply changes.
        """
        self._load_config()
        if self._is_connected:
            logger.info("Config reloaded. Disconnect and reconnect to apply changes.")


# Global singleton instance
mcp_manager = MCPClientManager()


def get_mcp_tools() -> List[Any]:
    """Convenience function to get MCP tools synchronously.
    
    Returns:
        List of LangChain-compatible tools from MCP servers
    """
    return mcp_manager.get_tools_sync()

