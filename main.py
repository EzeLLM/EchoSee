"""Main application entry point for EchoSee using MCP architecture."""

import asyncio
from typing import Dict, Any, List
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from tts.tts import TTS
from stt.stt import STT
from logger import logger as _logger
from mcp_servers.utils import load_mcp_server_configs

log = _logger.Logger("main")


class MCPOrchestrator:
    """Orchestrates communication with MCP servers."""
    
    def __init__(self):
        self.servers = load_mcp_server_configs()
        self.tts = TTS()
        self.stt = STT()
        log.info("MCP Orchestrator initialized")
    
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
    
    async def process_user_input(self, user_input: str) -> str:
        """Process user input and route to appropriate MCP servers."""
        log.info(f"Processing user input: {user_input}")
        
        # This is a simplified router - in a real implementation, 
        # you would use an LLM or pattern matching to determine intent
        lower_input = user_input.lower()
        
        try:
            # Time/date queries
            if any(word in lower_input for word in ["time", "date", "what day", "what month"]):
                if "time" in lower_input:
                    result = await self.call_tool("time", "get_current_time", {})
                    return f"The current time is {result['data']['time']}"
                else:
                    result = await self.call_tool("time", "get_current_date", {})
                    return f"Today's date is {result['data']['date']}"
            
            # Alarm management
            elif any(word in lower_input for word in ["alarm", "reminder", "wake me"]):
                if "stop" in lower_input or "cancel" in lower_input:
                    result = await self.call_tool("alarm", "stop_alarm", {})
                    return result['data']['message']
                elif "in" in lower_input:
                    # Simple parsing for demo - extract time delta
                    # In production, use proper NLP
                    result = await self.call_tool("alarm", "set_alarm_with_delta", {
                        "delta": "00:00:05:00",  # 5 minutes for demo
                        "label": "User requested alarm"
                    })
                    return f"Alarm set for {result['data']['scheduled_time']}"
            
            # Search queries
            elif any(word in lower_input for word in ["search", "find", "look up", "what is", "tell me about"]):
                # Determine if robust search is needed
                if "detailed" in lower_input or "everything" in lower_input:
                    result = await self.call_tool("search", "robust_search", {
                        "query": user_input,
                        "max_queries": 3
                    })
                    return result['data']['answer']
                else:
                    result = await self.call_tool("search", "simple_search", {
                        "query": user_input,
                        "max_results": 5
                    })
                    if result['data']['results']:
                        first_result = result['data']['results'][0]
                        return f"{first_result['title']}: {first_result['content']}"
                    return "No results found."
            
            # LeetCode queries
            elif "leetcode" in lower_input:
                if "daily" in lower_input:
                    result = await self.call_tool("leetcode", "leetcode_daily", {})
                else:
                    # Extract problem number/slug from input
                    result = await self.call_tool("leetcode", "leetcode_problem", {
                        "slug": "two-sum"  # Demo default
                    })
                return f"LeetCode Problem: {result['data'].get('title', 'Unknown')}"
            
            # Notification sending
            elif any(word in lower_input for word in ["notify", "send notification", "alert"]):
                result = await self.call_tool("notify", "notify_send", {
                    "channel": "me",
                    "message": user_input,
                    "priority": "normal"
                })
                return "Notification sent successfully!"
            
            # Default response
            else:
                return "I'm not sure how to help with that. Try asking about time, alarms, searches, or notifications."
                
        except Exception as e:
            log.error(f"Error processing request: {e}")
            return f"Sorry, I encountered an error: {str(e)}"


async def main():
    """Main application loop."""
    print("EchoSee MCP Application Started!")
    print("Type 'quit' to exit\n")
    
    orchestrator = MCPOrchestrator()
    
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
