from langchain_openai import ChatOpenAI
from agent_management import agents
import os
import dotenv
from typing import List, Dict
dotenv.load_dotenv()
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, BaseMessage
from utils.utils import llm, llm_config


class AgentManager:
    def __init__(self):
        # Set up tools
        self.tools = [agents.search, agents.code_agent, agents.get_current_time, agents.get_current_date]
        
        # Initialize conversation history
        self.conversation_history: List[Dict[str, List[BaseMessage]]] = []
        
        # Maintain a running list of messages for the agent
        self.agent_messages: List[BaseMessage] = []
        
        # Create React agent
        self.prompt = llm_config["system_prompt"]
        self.agent = create_react_agent(llm, tools=self.tools, prompt=self.prompt)
    
    def process_message(self, message: str) -> List[BaseMessage]:
        """
        Process a message and maintain conversation history.
        
        Args:
            message: The user's input message
            
        Returns:
            List of response messages
        """
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
        
        return response["messages"]
    
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


if __name__ == "__main__":
    # Test the agent manager
    agent_manager = AgentManager()
    
    # Test with a sample message
    response_messages = agent_manager.process_message("Hey jarvis what is the current time and date?")
    for m in response_messages:
        m.pretty_print()
    
    # Print conversation history
    print("\nConversation History:")
    for turn in agent_manager.get_conversation_history():
        print("Input:", turn["input"][0].content)
        print("Output:", turn["output"][-1].content)
        print("---")

