import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- Configuration ---
load_dotenv(dotenv_path="studio/.env")

if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
   raise ValueError("OPENAI_API_KEY and TAVILY_API_KEY must be set in 05_tool_calling/studio/.env")

# --- 1. Define Tools ---
tools = [TavilySearchResults(max_results=2)]

# --- 2. Define the State ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- 3. Define the Nodes ---
llm = ChatOpenAI(model="gpt-4o")
model_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    """Invokes the LLM to get the next action or a final response."""
    print("--- Calling Agent Node ---")
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

# --- 4. Define the Conditional Edge ---
def should_continue(state: AgentState) -> str:
    """Determines the next step after the agent node has been called."""
    print("--- Checking Condition ---")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print("   Condition: Tool call detected. Routing to tools.")
        return "tools"
    print("   Condition: No tool call. Ending.")
    return END

# --- 5. Build and Compile the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent")
app = workflow.compile()
