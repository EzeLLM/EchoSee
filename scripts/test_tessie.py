#!/usr/bin/env python3
"""
Tessie MCP Server Test Script
=============================
Test script for Tesla vehicle control via Tessie MCP server.

Usage:
    python -m scripts.test_tessie                    # Test connection & list tools
    python -m scripts.test_tessie --start-climate   # Start climate control
    python -m scripts.test_tessie --stop-climate    # Stop climate control
    python -m scripts.test_tessie --status          # Get vehicle status
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dotenv
dotenv.load_dotenv()


# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'
    
    SUCCESS = GREEN
    ERROR = RED
    WARNING = YELLOW
    INFO = BLUE
    ACCENT = CYAN


class Symbols:
    CHECK = '✓'
    CROSS = '✗'
    ARROW = '→'
    CAR = '🚗'
    TEMP = '🌡️'
    SNOW = '❄️'
    SUN = '☀️'
    PLUG = '🔌'
    BATTERY = '🔋'
    GEAR = '⚙'
    ROCKET = '🚀'
    WARNING = '⚠'
    INFO = 'ℹ'


def print_header():
    """Print beautiful ASCII header."""
    header = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   {Colors.HEADER}████████╗███████╗███████╗███████╗██╗███████╗{Colors.CYAN}                   ║
║   {Colors.HEADER}╚══██╔══╝██╔════╝██╔════╝██╔════╝██║██╔════╝{Colors.CYAN}                   ║
║   {Colors.HEADER}   ██║   █████╗  ███████╗███████╗██║█████╗{Colors.CYAN}                     ║
║   {Colors.HEADER}   ██║   ██╔══╝  ╚════██║╚════██║██║██╔══╝{Colors.CYAN}                     ║
║   {Colors.HEADER}   ██║   ███████╗███████║███████║██║███████╗{Colors.CYAN}                   ║
║   {Colors.HEADER}   ╚═╝   ╚══════╝╚══════╝╚══════╝╚═╝╚══════╝{Colors.CYAN}                   ║
║                                                                  ║
║         {Colors.DIM}Tesla Vehicle Control via MCP{Colors.CYAN}{Colors.BOLD}                      ║
║                        {Colors.DIM}EchoSee{Colors.CYAN}{Colors.BOLD}                                  ║
╚══════════════════════════════════════════════════════════════════╝
{Colors.END}"""
    print(header)


def print_section(title: str, icon: str = Symbols.GEAR):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{icon} {title}{Colors.END}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.END}")


def print_success(message: str):
    """Print success message."""
    print(f"  {Colors.SUCCESS}{Symbols.CHECK}{Colors.END} {message}")


def print_error(message: str):
    """Print error message."""
    print(f"  {Colors.ERROR}{Symbols.CROSS}{Colors.END} {message}")


def print_info(label: str, value: str):
    """Print info line."""
    print(f"  {Colors.DIM}{label}:{Colors.END} {Colors.ACCENT}{value}{Colors.END}")


async def get_tessie_client():
    """Create MCP client for Tessie server."""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        print_error("langchain-mcp-adapters not installed")
        print(f"  {Colors.DIM}Install with: pip install langchain-mcp-adapters{Colors.END}")
        return None
    
    # Tessie server config
    config = {
        "tesla": {
            "transport": "sse",
            "url": "http://0.0.0.0:8000/sse",
        }
    }
    
    return MultiServerMCPClient(config)


async def test_connection():
    """Test connection to Tessie MCP server and list available tools."""
    print_section("Testing Tessie Connection", Symbols.PLUG)
    
    client = await get_tessie_client()
    if not client:
        return False
    
    try:
        print(f"  {Colors.DIM}Connecting to Tessie MCP server...{Colors.END}")
        tools = await client.get_tools()
        
        print_success(f"Connected! Found {len(tools)} tools")
        
        print(f"\n  {Colors.BOLD}Available Tools:{Colors.END}")
        
        # Group tools by category
        climate_tools = []
        battery_tools = []
        vehicle_tools = []
        other_tools = []
        
        for tool in tools:
            name = tool.name.lower()
            if 'climate' in name or 'temp' in name or 'heater' in name:
                climate_tools.append(tool)
            elif 'battery' in name or 'charge' in name or 'energy' in name:
                battery_tools.append(tool)
            elif 'vehicle' in name or 'status' in name or 'location' in name or 'door' in name or 'lock' in name:
                vehicle_tools.append(tool)
            else:
                other_tools.append(tool)
        
        if climate_tools:
            print(f"\n    {Colors.CYAN}{Symbols.TEMP} Climate Controls:{Colors.END}")
            for tool in climate_tools:
                print(f"      {Colors.DIM}•{Colors.END} {tool.name}")
        
        if battery_tools:
            print(f"\n    {Colors.GREEN}{Symbols.BATTERY} Battery & Charging:{Colors.END}")
            for tool in battery_tools:
                print(f"      {Colors.DIM}•{Colors.END} {tool.name}")
        
        if vehicle_tools:
            print(f"\n    {Colors.BLUE}{Symbols.CAR} Vehicle Status:{Colors.END}")
            for tool in vehicle_tools:
                print(f"      {Colors.DIM}•{Colors.END} {tool.name}")
        
        if other_tools:
            print(f"\n    {Colors.YELLOW}{Symbols.GEAR} Other:{Colors.END}")
            for tool in other_tools:
                print(f"      {Colors.DIM}•{Colors.END} {tool.name}")
        
        return True
        
    except Exception as e:
        print_error(f"Connection failed: {e}")
        return False


async def invoke_tool(tool_name: str, args: dict = None):
    """Invoke a specific tool on the Tessie server."""
    client = await get_tessie_client()
    if not client:
        return None
    
    try:
        tools = await client.get_tools()
        
        # Find the tool
        tool = None
        for t in tools:
            if t.name == tool_name:
                tool = t
                break
        
        if not tool:
            print_error(f"Tool '{tool_name}' not found")
            return None
        
        # Invoke the tool
        result = await tool.ainvoke(args or {})
        return result
        
    except Exception as e:
        print_error(f"Tool invocation failed: {e}")
        return None


async def start_climate(temperature: float = None):
    """Start climate control / preconditioning."""
    print_section("Starting Climate Control", Symbols.SUN)
    
    print(f"  {Colors.DIM}Sending start_climate command...{Colors.END}")
    
    args = {}
    if temperature:
        args['temperature'] = temperature
        print_info("Target temperature", f"{temperature}°C")
    
    result = await invoke_tool("start_climate", args)
    
    if result:
        print_success("Climate control started!")
        print(f"\n  {Colors.BOLD}Response:{Colors.END}")
        print(f"  {Colors.DIM}{result}{Colors.END}")
        return True
    else:
        print_error("Failed to start climate control")
        return False


async def stop_climate():
    """Stop climate control / preconditioning."""
    print_section("Stopping Climate Control", Symbols.SNOW)
    
    print(f"  {Colors.DIM}Sending stop_climate command...{Colors.END}")
    
    result = await invoke_tool("stop_climate", {})
    
    if result:
        print_success("Climate control stopped!")
        print(f"\n  {Colors.BOLD}Response:{Colors.END}")
        print(f"  {Colors.DIM}{result}{Colors.END}")
        return True
    else:
        print_error("Failed to stop climate control")
        return False


async def set_temperature(temperature: float):
    """Set cabin temperature."""
    print_section(f"Setting Temperature to {temperature}°C", Symbols.TEMP)
    
    print(f"  {Colors.DIM}Sending set_temperature command...{Colors.END}")
    
    result = await invoke_tool("set_temperature", {"temperature": temperature})
    
    if result:
        print_success(f"Temperature set to {temperature}°C!")
        print(f"\n  {Colors.BOLD}Response:{Colors.END}")
        print(f"  {Colors.DIM}{result}{Colors.END}")
        return True
    else:
        print_error("Failed to set temperature")
        return False


async def get_vehicle_status():
    """Get comprehensive vehicle status."""
    print_section("Vehicle Status", Symbols.CAR)
    
    status_tools = [
        ("get_vehicle_status", "Vehicle State"),
        ("get_battery_level", "Battery Level"),
        ("get_is_climate_on", "Climate Status"),
        ("get_outside_temp", "Outside Temperature"),
        ("get_location", "Location"),
    ]
    
    for tool_name, label in status_tools:
        try:
            result = await invoke_tool(tool_name, {})
            if result is not None:
                print_info(label, str(result))
            else:
                print(f"  {Colors.DIM}{label}:{Colors.END} {Colors.WARNING}unavailable{Colors.END}")
        except Exception as e:
            print(f"  {Colors.DIM}{label}:{Colors.END} {Colors.ERROR}error{Colors.END}")


async def get_climate_status():
    """Get detailed climate status."""
    print_section("Climate Status", Symbols.TEMP)
    
    climate_tools = [
        ("get_is_climate_on", "Climate Active"),
        ("get_outside_temp", "Outside Temp"),
        ("get_seat_heater_left", "Driver Seat Heater"),
        ("get_seat_heater_right", "Passenger Seat Heater"),
        ("get_steering_wheel_heater", "Steering Wheel Heater"),
        ("get_battery_heater_on", "Battery Heater"),
    ]
    
    for tool_name, label in climate_tools:
        try:
            result = await invoke_tool(tool_name, {})
            if result is not None:
                # Format boolean values nicely
                if isinstance(result, bool):
                    value = f"{Colors.GREEN}ON{Colors.END}" if result else f"{Colors.DIM}off{Colors.END}"
                elif isinstance(result, (int, float)):
                    value = f"{result}°C" if 'temp' in tool_name.lower() else str(result)
                else:
                    value = str(result)
                print(f"  {Colors.DIM}{label}:{Colors.END} {value}")
            else:
                print(f"  {Colors.DIM}{label}:{Colors.END} {Colors.WARNING}unavailable{Colors.END}")
        except Exception as e:
            print(f"  {Colors.DIM}{label}:{Colors.END} {Colors.ERROR}error{Colors.END}")


async def get_battery_status():
    """Get battery and charging status."""
    print_section("Battery & Charging", Symbols.BATTERY)
    
    result = await invoke_tool("get_battery_summary", {})
    if result:
        print(f"  {result}")
    else:
        # Fallback to individual tools
        battery_tools = [
            ("get_battery_level", "Battery Level"),
            ("get_charging_state", "Charging State"),
            ("get_charge_limit_soc", "Charge Limit"),
            ("get_minutes_to_full_charge", "Time to Full"),
            ("get_energy_remaining", "Energy Remaining"),
        ]
        
        for tool_name, label in battery_tools:
            try:
                result = await invoke_tool(tool_name, {})
                if result is not None:
                    if 'level' in tool_name or 'limit' in tool_name:
                        value = f"{result}%"
                    elif 'minutes' in tool_name:
                        value = f"{result} min"
                    elif 'energy' in tool_name:
                        value = f"{result} kWh"
                    else:
                        value = str(result)
                    print_info(label, value)
            except:
                pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test Tessie MCP server for Tesla vehicle control.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                         Test connection and list tools
  %(prog)s --start-climate         Start climate control
  %(prog)s --start-climate --temp 22  Start climate at 22°C
  %(prog)s --stop-climate          Stop climate control
  %(prog)s --status                Get vehicle status
  %(prog)s --climate-status        Get climate status
  %(prog)s --battery               Get battery status
        """
    )
    
    parser.add_argument(
        '--start-climate',
        action='store_true',
        help='Start climate control / preconditioning'
    )
    parser.add_argument(
        '--stop-climate',
        action='store_true',
        help='Stop climate control'
    )
    parser.add_argument(
        '--temp', '--temperature',
        type=float,
        help='Set target temperature in Celsius'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Get vehicle status'
    )
    parser.add_argument(
        '--climate-status',
        action='store_true',
        help='Get detailed climate status'
    )
    parser.add_argument(
        '--battery',
        action='store_true',
        help='Get battery and charging status'
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
    
    print_header()
    
    async def run():
        # Always test connection first
        connected = await test_connection()
        
        if not connected:
            print(f"\n  {Colors.ERROR}Cannot proceed - server not connected{Colors.END}")
            return 1
        
        # Handle specific commands
        if args.start_climate:
            await start_climate(args.temp)
        elif args.stop_climate:
            await stop_climate()
        elif args.temp and not args.start_climate:
            await set_temperature(args.temp)
        elif args.status:
            await get_vehicle_status()
        elif args.climate_status:
            await get_climate_status()
        elif args.battery:
            await get_battery_status()
        
        return 0
    
    exit_code = asyncio.run(run())
    print()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()


