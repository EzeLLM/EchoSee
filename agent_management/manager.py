from langchain_openai import ChatOpenAI
from agent_management import agents
import os
import dotenv
dotenv.load_dotenv()
from langgraph.graph import START, END, StateGraph
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

# Set up LLM and agent
llm = ChatOpenAI(api_key=os.environ['OPENAI_API_KEY'], model='gpt-4o')
llm_agent = llm.bind_tools([agents.search])

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_agent.invoke(state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([agents.search]))  # Use ToolNode with your search tool
builder.add_edge(START, "tool_calling_llm")

# The key part - use tools_condition without providing a mapping dictionary
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,  # This automatically routes to "tools" node when tool calls present
)

# After tools, route back to the LLM
builder.add_edge("tools", "tool_calling_llm")
graph = builder.compile()


from langchain_core.messages import HumanMessage
messages = [HumanMessage(content="who won the most recent american presidential election?")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()

