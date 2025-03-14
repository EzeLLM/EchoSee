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
    messages = [HumanMessage(content="calculate the 4th factorial of e^12 then divide it by 12 and give me the result")]
    messages = graph.invoke({"messages": messages})
    for m in messages['messages']:
        m.pretty_print()

