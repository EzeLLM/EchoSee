"""Main application entry point for EchoSee using MCP architecture."""

import asyncio
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from tts.tts import TTS
from stt.stt import STT
from logger import logger as _logger
from mcp_servers.utils import load_mcp_server_configs
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

log = _logger.Logger("main")

@dataclass
class ToolInfo:
    """Information about an available tool."""
    name: str
    server: str
    description: str
    schema: Dict[str, Any]


class MCPOrchestrator:
    """Orchestrates communication with MCP servers."""
    
    def __init__(self):
        self.servers = load_mcp_server_configs()
        self.tts = TTS()
        self.stt = STT()
        self.available_tools: Dict[str, ToolInfo] = {}
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        log.info("MCP Orchestrator initialized")
    
    async def discover_tools(self, timeout_per_server: int = 5, max_retries: int = 3):
        """Discover all available tools from all MCP servers."""
        log.info("Discovering tools from MCP servers...")
        
        for server_name, server_info in self.servers.items():
            url = server_info["url"]
            log.info(f"Attempting to connect to {server_name} at {url}")
            
            success = False
            for attempt in range(max_retries):
                try:
                    # Add timeout to prevent hanging
                    async with asyncio.timeout(timeout_per_server):
                        async with streamablehttp_client(url) as (r, w, _):
                            session = ClientSession(r, w)
                            await session.initialize()
                            
                            # Get list of tools
                            tools_result = await session.list_tools()
                            
                            for tool in tools_result.tools:
                                try:
                                    # Handle different SDK attribute names
                                    schema = (
                                        getattr(tool, "inputSchema", None)
                                        or getattr(tool, "input_schema", None)
                                        or {}
                                    )
                                    description = getattr(tool, "description", None) or f"Tool {tool.name} from {server_name}"
                                    tool_info = ToolInfo(
                                        name=tool.name,
                                        server=server_name,
                                        description=description,
                                        schema=schema,
                                    )
                                    self.available_tools[tool.name] = tool_info
                                    log.info(f"✓ Discovered tool: {tool.name} from {server_name}")
                                except Exception as inner_exc:
                                    log.warning(
                                        f"Failed to register tool from {server_name}: {getattr(tool, 'name', '<unknown>')} - {inner_exc}"
                                    )
                            
                            log.info(f"✓ Successfully connected to {server_name} ({len(tools_result.tools)} tools)")
                            success = True
                            break
                            
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        log.info(f"Timeout connecting to {server_name}, retrying in 2s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(2)
                    else:
                        log.warning(f"✗ Timeout connecting to {server_name} at {url} after {max_retries} attempts")
                except Exception as e:
                    if attempt < max_retries - 1:
                        log.info(f"Error connecting to {server_name}: {e}, retrying in 2s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(2)
                    else:
                        log.warning(f"✗ Failed to connect to {server_name} at {url} after {max_retries} attempts: {e}")
            
            if not success:
                log.warning(f"✗ Could not connect to {server_name} after {max_retries} attempts")
        
        log.info(f"Tool discovery complete. Found {len(self.available_tools)} tools from {len(self.servers)} servers.")
    
    async def call_tool(self, server_name: str, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on a specific MCP server."""
        server_info = self.servers.get(server_name)
        if not server_info:
            raise ValueError(f"Unknown server: {server_name}")
        
        url = server_info["url"]
        try:
            async with streamablehttp_client(url) as (r, w, _):
                session = ClientSession(r, w)
                await session.initialize()
                result = await session.call_tool(tool_name, args)
                return result.content
        except Exception as e:
            log.error(f"Error calling {tool_name} on {server_name}: {e}")
            raise
    
    async def determine_tool_and_args(self, user_input: str) -> Optional[tuple[str, Dict[str, Any]]]:
        """Use LLM to determine which tool to use and what arguments to pass."""
        
        # Create a tool catalog for the LLM
        tool_catalog = []
        for tool_name, tool_info in self.available_tools.items():
            tool_catalog.append({
                "name": tool_name,
                "server": tool_info.server,
                "description": tool_info.description,
                "schema": tool_info.schema
            })
        
        system_prompt = f"""You are an intelligent tool router for an MCP-based assistant system.

Given a user request, you need to determine:
1. Which tool to use from the available tools
2. What arguments to pass to that tool

Available tools:
{json.dumps(tool_catalog, indent=2)}

Rules:
- Respond with EXACTLY this JSON format: {{"tool": "tool_name", "args": {{"param": "value"}}}}
- If no tool matches, respond with: {{"tool": null, "args": {{}}}}
- Extract actual values from the user's request for the arguments
- Use common sense to map user requests to appropriate tools
- For time/date requests, use time tools
- For search requests, use search tools (robust_search for detailed/complex queries, simple_search for basic ones)
- For alarms/reminders, use alarm tools
- For notifications, use notify tools
- For LeetCode, use leetcode tools

Examples:
User: "What time is it?" -> {{"tool": "get_current_time", "args": {{}}}}
User: "Search for Python tutorials" -> {{"tool": "simple_search", "args": {{"query": "Python tutorials", "max_results": 5}}}}
User: "Set an alarm in 10 minutes" -> {{"tool": "set_alarm_with_delta", "args": {{"delta": "00:00:10:00", "label": "User alarm"}}}}
User: "Send me a notification about dinner" -> {{"tool": "notify_send", "args": {{"channel": "me", "message": "dinner", "priority": "normal"}}}}
"""


        user_prompt = f"User request: '{user_input}'"
        
        try:
            response = await asyncio.to_thread(
                lambda: self.llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ])
            )
            
            # Parse the JSON response
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
            
            parsed = json.loads(response_text)
            tool_name = parsed.get("tool")
            args = parsed.get("args", {})
            
            if tool_name and tool_name in self.available_tools:
                return tool_name, args
            
            return None
            
        except Exception as e:
            log.error(f"Error determining tool: {e}")
            return None

    async def process_user_input(self, user_input: str) -> str:
        """Process user input using LLM-based tool routing."""
        log.info(f"Processing user input: {user_input}")
        
        if not self.available_tools:
            return "System initializing... Please try again in a moment."
        
        try:
            # Use LLM to determine tool and arguments
            tool_result = await self.determine_tool_and_args(user_input)
            
            if not tool_result:
                # Try to help the user understand what's available
                available_categories = set(tool_info.server for tool_info in self.available_tools.values())
                return f"I couldn't understand your request. I can help with: {', '.join(available_categories)}. Available tools: {', '.join(self.available_tools.keys())}"
            
            tool_name, args = tool_result
            tool_info = self.available_tools[tool_name]
            
            # Call the tool
            log.info(f"Calling tool {tool_name} on server {tool_info.server} with args: {args}")
            result = await self.call_tool(tool_info.server, tool_name, args)
            
            # Format the response based on the result
            if isinstance(result, dict):
                if result.get("ok"):
                    data = result.get("data", {})
                    
                    # Handle different response types
                    if "time" in data:
                        return f"The current time is {data['time']}"
                    elif "date" in data:
                        return f"Today's date is {data['date']}"
                    elif "answer" in data:
                        return data["answer"]
                    elif "results" in data and data["results"]:
                        first_result = data["results"][0]
                        return f"{first_result.get('title', 'Result')}: {first_result.get('content', 'No content')}"
                    elif "message" in data:
                        return data["message"]
                    elif "alarm_id" in data:
                        return f"Alarm set successfully! ID: {data['alarm_id']}"
                    elif "title" in data:
                        return f"LeetCode Problem: {data['title']}"
                    else:
                        return f"Task completed successfully. Result: {data}"
                else:
                    error = result.get("error", {})
                    return f"Error: {error.get('message', 'Unknown error occurred')}"
            else:
                return str(result)
                
        except Exception as e:
            log.error(f"Error processing request: {e}")
            return f"Sorry, I encountered an error: {str(e)}"


async def main():
    """Main application loop."""
    print("EchoSee MCP Application Started!")
    print("Note: Make sure to run './launch.sh' first to start MCP servers!")
    print("Initializing and discovering tools...")
    
    orchestrator = MCPOrchestrator()
    
    # Discover all available tools from MCP servers
    await orchestrator.discover_tools()
    
    if orchestrator.available_tools:
        print(f"\n✓ Ready! Discovered {len(orchestrator.available_tools)} tools from MCP servers.")
        print("Available tools:", ", ".join(orchestrator.available_tools.keys()))
    else:
        print("\n⚠️  No tools discovered. Make sure MCP servers are running with './launch.sh'")
        print("You can still try to use the system, but functionality will be limited.")
    
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ")
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            # Process through MCP orchestrator
            response = await orchestrator.process_user_input(user_input)
            
            print(f"\nAssistant: {response}\n")
            
            # Optional: TTS output
            # orchestrator.tts.play_with_device(response, device=tts_config['device'])
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            log.error(f"Unhandled error in main loop: {e}")
            print(f"An error occurred: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())
