from langchain_openai import ChatOpenAI
import os
import dotenv
import logging
from typing import List, Dict, Optional, Generator
dotenv.load_dotenv()
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, BaseMessage
from utils.utils import llm
from langchain_core.tools import tool
import threading
from core.config_manager import config
from core.conversation_storage import ConversationStorage

# Import tools from new organized structure
from agent_management.tools import time_tools, search_tools, event_tools, notification_tools, code_tools

logger = logging.getLogger(__name__)


class AgentManager:
    def __init__(self, enable_persistence: bool = None):
        """Initialize AgentManager.

        Args:
            enable_persistence: Whether to enable conversation persistence.
                               If None, uses value from config (defaults to True).
        """
        # Set up tools
        self.config = config.get_section('AgentManager')

        # Initialize conversation storage
        if enable_persistence is None:
            enable_persistence = self.config.get('persistence_enabled', True)

        self.storage: Optional[ConversationStorage] = None
        if enable_persistence:
            self.storage = ConversationStorage()
            logger.info("Conversation persistence enabled")
        self.tools = [
            # Time tools
            time_tools.get_current_time,
            time_tools.get_current_date,
            # Search tools
            search_tools.search,
            search_tools.robust_search,
            # Event tools
            event_tools.set_alarm_at_specific_time,
            event_tools.set_alarm_with_time_delta,
            event_tools.stop_alarm,
            # Notification tools
            notification_tools.send_notification,
            # Code tools
            code_tools.leetcode_agent,
        ]
        
        # Initialize conversation history
        self.conversation_history: List[Dict[str, List[BaseMessage]]] = []
        
        # Maintain a running list of messages for the agent
        self.agent_messages: List[BaseMessage] = []
        
        # Load recent context from storage if available
        max_context_turns = self.config.get('max_context_turns', 5)
        if self.storage and max_context_turns > 0:
            self._load_context_from_storage(max_context_turns)
        
        # Create React agent
        llm_section = config.get_section('LLM')
        self.prompt = llm_section["system_prompt"]
        self.agent = create_react_agent(llm, tools=self.tools, prompt=self.prompt)
        self.clear_timer = None

    def _load_context_from_storage(self, max_turns: int):
        """Load recent conversation context from storage.

        Args:
            max_turns: Maximum number of past turns to load
        """
        try:
            recent = self.storage.get_recent_history(limit=max_turns)
            if recent:
                logger.info(f"Loading {len(recent)} past conversation turns")
                for turn in recent:
                    self.agent_messages.append(
                        HumanMessage(content=turn['user_message'])
                    )
                    # Create AI message with the assistant response
                    from langchain_core.messages import AIMessage
                    self.agent_messages.append(
                        AIMessage(content=turn['assistant_message'])
                    )
        except Exception as e:
            logger.warning(f"Could not load context from storage: {e}")
    def _reset_history_timer(self):
        if self.clear_timer is not None:
            self.clear_timer.cancel()
        self.clear_timer = threading.Timer(60*(int(self.config['clear_time'])), self.clear_history)
        self.clear_timer.daemon = True
        self.clear_timer.start()

    def process_message(self, message: str) -> List[BaseMessage]:
        """Process a message and maintain conversation history.
        
        Args:
            message: The user's input message
            
        Returns:
            List of response messages
        """
        # Reset history timer
        self._reset_history_timer()
        # Add current message to the running list
        current_message = HumanMessage(content=message)
        self.agent_messages.append(current_message)
        
        # Get response from agent using maintained message list
        response = self.agent.invoke({"messages": self.agent_messages})
        
        # Add agent responses to running message list
        self.agent_messages.extend(response["messages"])
        
        # Store only this turn in history
        self.conversation_history.append({
            "input": [current_message],
            "output": response["messages"]
        })

        # Save to persistent storage
        if self.storage:
            try:
                # Get the final assistant message
                assistant_msg = response["messages"][-1].content if response["messages"] else ""
                
                # Collect metadata about tool usage
                tool_calls = []
                for msg in response["messages"]:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        tool_calls.extend([tc.get('name', 'unknown') for tc in msg.tool_calls])
                
                metadata = {"tool_calls": tool_calls} if tool_calls else None
                self.storage.save_turn(message, assistant_msg, metadata)
            except Exception as e:
                logger.warning(f"Could not save conversation turn: {e}")

        return response["messages"]

    def process_message_streaming(self, message: str) -> Generator[str, None, None]:
        """Process a message with streaming response.

        Yields text chunks as they are generated by the LLM, enabling
        real-time TTS playback. Falls back to non-streaming if needed.

        Args:
            message: The user's input message

        Yields:
            Text chunks from the assistant's response
        """
        from langchain_core.messages import AIMessage, AIMessageChunk

        # Reset history timer
        self._reset_history_timer()

        # Add current message to the running list
        current_message = HumanMessage(content=message)
        self.agent_messages.append(current_message)

        full_response_text = ""
        all_response_messages = []
        tool_calls = []
        yielded_anything = False

        try:
            # Stream response from agent using stream_mode for token streaming
            for event in self.agent.stream({"messages": self.agent_messages}, stream_mode="updates"):
                # LangGraph returns updates per node
                # Each event is a dict with node name as key
                for node_name, node_output in event.items():
                    if node_name == "agent" and "messages" in node_output:
                        for msg in node_output["messages"]:
                            all_response_messages.append(msg)

                            # Track tool calls
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                tool_calls.extend([tc.get('name', 'unknown') for tc in msg.tool_calls])

                            # Stream AI message content
                            if isinstance(msg, (AIMessage, AIMessageChunk)):
                                if hasattr(msg, 'content') and msg.content:
                                    # Skip empty content or tool-call-only messages
                                    if msg.content.strip():
                                        content = msg.content
                                        full_response_text += content
                                        yielded_anything = True
                                        yield content

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            # Fall back to non-streaming
            logger.info("Falling back to non-streaming mode")
            response = self.agent.invoke({"messages": self.agent_messages})
            if "messages" in response:
                all_response_messages = response["messages"]
                if all_response_messages:
                    content = all_response_messages[-1].content
                    full_response_text = content
                    yield content
                    yielded_anything = True

        # If streaming yielded nothing, fall back to invoke
        if not yielded_anything:
            logger.warning("Streaming yielded no content, falling back to invoke")
            # Remove the message we added (will be re-added by invoke path)
            self.agent_messages.pop()
            response = self.agent.invoke({"messages": self.agent_messages + [current_message]})
            if "messages" in response:
                all_response_messages = response["messages"]
                self.agent_messages.append(current_message)
                self.agent_messages.extend(all_response_messages)
                if all_response_messages:
                    content = all_response_messages[-1].content
                    full_response_text = content
                    yield content
        else:
            # Add all responses to message history
            self.agent_messages.extend(all_response_messages)

        # Store in conversation history
        self.conversation_history.append({
            "input": [current_message],
            "output": all_response_messages
        })

        # Save to persistent storage
        if self.storage:
            try:
                metadata = {"tool_calls": tool_calls, "streaming": True} if tool_calls else {"streaming": True}
                self.storage.save_turn(message, full_response_text, metadata)
            except Exception as e:
                logger.warning(f"Could not save conversation turn: {e}")

    def get_conversation_history(self) -> List[Dict[str, List[BaseMessage]]]:
        """
        Get the full conversation history.
        
        Returns:
            List of conversation turns, each containing input and output messages
        """
        return self.conversation_history
    def clear_history(self):
        """Clear the conversation history and running message list."""
        self.conversation_history = []
        self.agent_messages = []

    def search_past_conversations(self, query: str, limit: int = 10) -> List[Dict]:
        """Search past conversations by text.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching conversation turns
        """
        if not self.storage:
            return []
        return self.storage.search_conversations(query, limit)

    def get_conversation_stats(self) -> Dict:
        """Get conversation statistics.
        
        Returns:
            Dictionary with conversation stats
        """
        if not self.storage:
            return {"persistence_enabled": False}
        
        return {
            "persistence_enabled": True,
            "total_conversations": self.storage.get_conversation_count(),
            "current_session_turns": len(self.conversation_history),
            "loaded_context_messages": len(self.agent_messages)
        }


if __name__ == "__main__":
    # Test the agent manager
    agent_manager = AgentManager()
    
    # Test with a sample message
    response_messages = agent_manager.process_message("hey bro how are you doing, can you help me with leetcode problem 104. i just need some hints no need to solve it")
    for m in response_messages:
        m.pretty_print()
    
    # Print conversation history
    print("\nConversation History:")
    for turn in agent_manager.get_conversation_history():
        print("Input:", turn["input"][0].content)
        print("Output:", turn["output"][-1].content)
        print("---")
    import time 
    time.sleep(65)

