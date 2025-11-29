#!/usr/bin/env python3
"""
MCP Server Test Script
======================
A beautiful CLI tool to test MCP server connections and display loaded tools.

Usage:
    python -m scripts.test_mcp          # Test all configured servers
    python -m scripts.test_mcp --server <name>  # Test specific server
    python -m scripts.test_mcp --list   # List configured servers
"""

import asyncio
import argparse
import sys
from typing import Dict, List, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Initialize environment
import dotenv
dotenv.load_dotenv()


# ANSI color codes for beautiful output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    # Semantic aliases
    SUCCESS = GREEN
    ERROR = RED
    WARNING = YELLOW
    INFO = BLUE
    ACCENT = CYAN


class Symbols:
    CHECK = '✓'
    CROSS = '✗'
    ARROW = '→'
    DOT = '●'
    DIAMOND = '◆'
    STAR = '★'
    GEAR = '⚙'
    PLUG = '🔌'
    TOOL = '🔧'
    BOX = '📦'
    ROCKET = '🚀'
    WARNING = '⚠'
    INFO = 'ℹ'


def print_header():
    """Print beautiful ASCII header."""
    header = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   {Colors.HEADER}███╗   ███╗ ██████╗██████╗    ████████╗███████╗███████╗████████╗{Colors.CYAN}   ║
║   {Colors.HEADER}████╗ ████║██╔════╝██╔══██╗   ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝{Colors.CYAN}   ║
║   {Colors.HEADER}██╔████╔██║██║     ██████╔╝      ██║   █████╗  ███████╗   ██║{Colors.CYAN}      ║
║   {Colors.HEADER}██║╚██╔╝██║██║     ██╔═══╝       ██║   ██╔══╝  ╚════██║   ██║{Colors.CYAN}      ║
║   {Colors.HEADER}██║ ╚═╝ ██║╚██████╗██║           ██║   ███████╗███████║   ██║{Colors.CYAN}      ║
║   {Colors.HEADER}╚═╝     ╚═╝ ╚═════╝╚═╝           ╚═╝   ╚══════╝╚══════╝   ╚═╝{Colors.CYAN}      ║
║                                                                  ║
║         {Colors.DIM}Model Context Protocol Server Validator{Colors.CYAN}{Colors.BOLD}               ║
║                        {Colors.DIM}EchoSee{Colors.CYAN}{Colors.BOLD}                                  ║
╚══════════════════════════════════════════════════════════════════╝
{Colors.END}"""
    print(header)


def print_section(title: str, icon: str = Symbols.GEAR):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{icon} {title}{Colors.END}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.END}")


def print_server_status(name: str, result: Dict[str, Any]):
    """Print formatted server test result."""
    status = result.get('status', 'unknown')
    transport = result.get('transport', 'unknown')
    
    # Status indicator
    if status == 'connected':
        status_icon = f"{Colors.SUCCESS}{Symbols.CHECK}{Colors.END}"
        status_text = f"{Colors.SUCCESS}Connected{Colors.END}"
    elif status == 'error':
        status_icon = f"{Colors.ERROR}{Symbols.CROSS}{Colors.END}"
        status_text = f"{Colors.ERROR}Failed{Colors.END}"
    else:
        status_icon = f"{Colors.WARNING}{Symbols.WARNING}{Colors.END}"
        status_text = f"{Colors.WARNING}{status.title()}{Colors.END}"
    
    # Server name and status
    print(f"\n  {status_icon} {Colors.BOLD}{name}{Colors.END}")
    print(f"     {Colors.DIM}Transport:{Colors.END} {Colors.ACCENT}{transport}{Colors.END}")
    
    if status == 'connected':
        tool_count = result.get('tool_count', 0)
        tools = result.get('tools', [])
        
        print(f"     {Colors.DIM}Status:{Colors.END} {status_text}")
        print(f"     {Colors.DIM}Tools loaded:{Colors.END} {Colors.GREEN}{tool_count}{Colors.END}")
        
        if tools:
            print(f"\n     {Colors.BOLD}Available Tools:{Colors.END}")
            for tool in tools:
                tool_name = tool.get('name', 'unknown')
                tool_desc = tool.get('description', 'No description')
                print(f"       {Colors.CYAN}{Symbols.TOOL}{Colors.END} {Colors.BOLD}{tool_name}{Colors.END}")
                print(f"         {Colors.DIM}{tool_desc}{Colors.END}")
    
    elif status == 'error':
        error = result.get('error', 'Unknown error')
        print(f"     {Colors.DIM}Status:{Colors.END} {status_text}")
        print(f"     {Colors.ERROR}Error: {error}{Colors.END}")


def print_summary(results: List[Dict[str, Any]]):
    """Print test summary."""
    total = len(results)
    connected = sum(1 for r in results if r.get('status') == 'connected')
    failed = sum(1 for r in results if r.get('status') == 'error')
    total_tools = sum(r.get('tool_count', 0) for r in results)
    
    print_section("Summary", Symbols.STAR)
    
    # Stats box
    print(f"""
  {Colors.BOLD}┌────────────────────────────────────────┐{Colors.END}
  {Colors.BOLD}│{Colors.END}  {Colors.DIM}Servers Tested:{Colors.END}    {Colors.BOLD}{total:>16}{Colors.END}     {Colors.BOLD}│{Colors.END}
  {Colors.BOLD}│{Colors.END}  {Colors.SUCCESS}Connected:{Colors.END}          {Colors.SUCCESS}{connected:>16}{Colors.END}     {Colors.BOLD}│{Colors.END}
  {Colors.BOLD}│{Colors.END}  {Colors.ERROR}Failed:{Colors.END}             {Colors.ERROR}{failed:>16}{Colors.END}     {Colors.BOLD}│{Colors.END}
  {Colors.BOLD}│{Colors.END}  {Colors.CYAN}Total Tools:{Colors.END}        {Colors.CYAN}{total_tools:>16}{Colors.END}     {Colors.BOLD}│{Colors.END}
  {Colors.BOLD}└────────────────────────────────────────┘{Colors.END}
""")
    
    # Final status
    if failed == 0 and connected > 0:
        print(f"  {Colors.SUCCESS}{Colors.BOLD}{Symbols.ROCKET} All servers connected successfully!{Colors.END}")
    elif connected > 0:
        print(f"  {Colors.WARNING}{Symbols.WARNING} Some servers failed to connect.{Colors.END}")
    else:
        print(f"  {Colors.ERROR}{Symbols.CROSS} No servers connected.{Colors.END}")


def print_no_servers_message():
    """Print helpful message when no servers are configured."""
    print(f"""
  {Colors.WARNING}{Symbols.INFO} No MCP servers configured{Colors.END}

  {Colors.DIM}To add MCP servers, edit {Colors.CYAN}config.yml{Colors.DIM} and add server definitions:{Colors.END}

  {Colors.CYAN}MCP:
    enabled: true
    servers:
      my_server:
        transport: stdio
        command: python
        args:
          - /path/to/my_server.py{Colors.END}

  {Colors.DIM}Or for HTTP servers:{Colors.END}

  {Colors.CYAN}MCP:
    enabled: true
    servers:
      api_server:
        transport: streamable_http
        url: http://localhost:8000/mcp{Colors.END}

  {Colors.DIM}See the MCP documentation for more transport options.{Colors.END}
""")


def print_mcp_disabled_message():
    """Print message when MCP is disabled."""
    print(f"""
  {Colors.WARNING}{Symbols.WARNING} MCP is disabled{Colors.END}

  {Colors.DIM}MCP support is currently disabled in configuration.
  
  To enable, set {Colors.CYAN}MCP.enabled: true{Colors.DIM} in {Colors.CYAN}config.yml{Colors.END}
""")


def print_missing_deps_message():
    """Print message when MCP dependencies are missing."""
    print(f"""
  {Colors.ERROR}{Symbols.CROSS} MCP dependencies not installed{Colors.END}

  {Colors.DIM}Install the required packages:{Colors.END}

  {Colors.CYAN}pip install langchain-mcp-adapters langgraph{Colors.END}

  {Colors.DIM}Then try again.{Colors.END}
""")


def list_servers():
    """List all configured MCP servers."""
    from core.mcp_client import mcp_manager, HAS_MCP
    
    print_header()
    print_section("Configured MCP Servers", Symbols.BOX)
    
    if not HAS_MCP:
        print_missing_deps_message()
        return 1
    
    if not mcp_manager.enabled:
        print_mcp_disabled_message()
        return 1
    
    if not mcp_manager.has_servers:
        print_no_servers_message()
        return 0
    
    server_info = mcp_manager.get_server_info()
    
    for name, info in server_info.items():
        transport = info.get('transport', 'unknown')
        
        print(f"\n  {Colors.CYAN}{Symbols.PLUG}{Colors.END} {Colors.BOLD}{name}{Colors.END}")
        print(f"     {Colors.DIM}Transport:{Colors.END} {Colors.ACCENT}{transport}{Colors.END}")
        
        if transport == 'stdio':
            print(f"     {Colors.DIM}Command:{Colors.END} {info.get('command', 'N/A')}")
        else:
            print(f"     {Colors.DIM}URL:{Colors.END} {info.get('url', 'N/A')}")
    
    print(f"\n  {Colors.DIM}Total: {len(server_info)} server(s) configured{Colors.END}\n")
    return 0


async def test_server(server_name: str):
    """Test a specific MCP server."""
    from core.mcp_client import mcp_manager, HAS_MCP
    
    print_header()
    print_section(f"Testing Server: {server_name}", Symbols.GEAR)
    
    if not HAS_MCP:
        print_missing_deps_message()
        return 1
    
    if not mcp_manager.enabled:
        print_mcp_disabled_message()
        return 1
    
    if server_name not in mcp_manager.server_names:
        print(f"\n  {Colors.ERROR}{Symbols.CROSS} Server '{server_name}' not found{Colors.END}")
        print(f"\n  {Colors.DIM}Available servers: {', '.join(mcp_manager.server_names) or 'none'}{Colors.END}\n")
        return 1
    
    result = await mcp_manager.test_server(server_name)
    print_server_status(server_name, result)
    
    if result.get('status') == 'connected':
        print(f"\n  {Colors.SUCCESS}{Symbols.CHECK} Server test passed!{Colors.END}\n")
        return 0
    else:
        print(f"\n  {Colors.ERROR}{Symbols.CROSS} Server test failed!{Colors.END}\n")
        return 1


async def test_all_servers():
    """Test all configured MCP servers."""
    from core.mcp_client import mcp_manager, HAS_MCP
    
    print_header()
    print_section("MCP Configuration", Symbols.GEAR)
    
    if not HAS_MCP:
        print_missing_deps_message()
        return 1
    
    # Check if MCP is enabled
    if not mcp_manager.enabled:
        print_mcp_disabled_message()
        return 1
    
    # Check for configured servers
    if not mcp_manager.has_servers:
        print_no_servers_message()
        return 0
    
    server_names = mcp_manager.server_names
    print(f"  {Colors.DIM}Found {len(server_names)} server(s) configured{Colors.END}")
    
    print_section("Testing Servers", Symbols.PLUG)
    
    # Test all servers
    results = await mcp_manager.test_all_servers()
    
    for result in results:
        print_server_status(result.get('server_name', 'unknown'), result)
    
    # Print summary
    print_summary(results)
    
    # Return exit code based on results
    failed = sum(1 for r in results if r.get('status') == 'error')
    return 1 if failed > 0 else 0


async def test_agent_integration():
    """Test that MCP tools integrate correctly with the agent."""
    from core.mcp_client import mcp_manager, HAS_MCP
    
    print_section("Agent Integration Test", Symbols.ROCKET)
    
    if not HAS_MCP:
        print(f"  {Colors.WARNING}{Symbols.WARNING} Skipping - MCP dependencies not installed{Colors.END}")
        return
    
    if not mcp_manager.enabled or not mcp_manager.has_servers:
        print(f"  {Colors.DIM}Skipping - No MCP servers configured{Colors.END}")
        return
    
    try:
        # Import AgentManager to test integration
        from agent_management.agent_manager import AgentManager
        
        print(f"  {Colors.DIM}Initializing AgentManager...{Colors.END}")
        am = AgentManager(enable_persistence=False)  # Disable persistence for test
        
        # Count MCP tools
        mcp_tools = [t for t in am.tools if hasattr(t, 'name') and not callable(getattr(t, '__self__', None))]
        builtin_count = len([
            t for t in am.tools 
            if hasattr(t, '__module__') and 'agent_management.tools' in str(getattr(t, '__module__', ''))
        ])
        
        total_tools = len(am.tools)
        
        print(f"\n  {Colors.SUCCESS}{Symbols.CHECK}{Colors.END} {Colors.BOLD}AgentManager initialized successfully{Colors.END}")
        print(f"     {Colors.DIM}Total tools:{Colors.END} {Colors.CYAN}{total_tools}{Colors.END}")
        print(f"     {Colors.DIM}Tool names:{Colors.END}")
        
        for tool in am.tools[:20]:  # Show first 20 tools
            tool_name = getattr(tool, 'name', str(tool))
            print(f"       {Colors.CYAN}{Symbols.TOOL}{Colors.END} {tool_name}")
        
        if len(am.tools) > 20:
            print(f"       {Colors.DIM}... and {len(am.tools) - 20} more{Colors.END}")
        
    except Exception as e:
        print(f"  {Colors.ERROR}{Symbols.CROSS} Integration test failed: {e}{Colors.END}")


def main():
    """Main entry point for the MCP test script."""
    parser = argparse.ArgumentParser(
        description='Test MCP server connections and list available tools.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    Test all configured MCP servers
  %(prog)s --server math      Test only the 'math' server
  %(prog)s --list             List all configured servers
  %(prog)s --agent            Also test agent integration
        """
    )
    
    parser.add_argument(
        '--server', '-s',
        type=str,
        help='Test a specific server by name'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all configured servers without testing'
    )
    parser.add_argument(
        '--agent', '-a',
        action='store_true',
        help='Also test agent integration with MCP tools'
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable colored output'
    )
    
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith('_'):
                setattr(Colors, attr, '')
    
    # Handle different modes
    if args.list:
        sys.exit(list_servers())
    
    async def run_tests():
        if args.server:
            exit_code = await test_server(args.server)
        else:
            exit_code = await test_all_servers()
        
        if args.agent:
            await test_agent_integration()
        
        return exit_code
    
    exit_code = asyncio.run(run_tests())
    print()  # Final newline
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

