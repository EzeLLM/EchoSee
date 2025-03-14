from langchain_openai import ChatOpenAI
from agent_management import agents
import os
import dotenv
dotenv.load_dotenv()
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from utils.utils import llm, llm_config

# Set up tools
tools = [agents.search, agents.code_agent]

# Create React agent
# You can customize the prompt if needed
prompt = llm_config["system_prompt"]
graph = create_react_agent(llm, tools=tools, prompt=prompt)

if __name__ == "__main__":
    print(graph.get_graph().draw_mermaid())
    
    # Test the agent
    messages = [HumanMessage(content="You hvae the oppurtunity to save one person on earth. You can save one person by either saving them from a fire or saving them from a car accident. Which one would you save?")]
    messages = graph.invoke({"messages": messages})
    for m in messages['messages']:
        m.pretty_print()

