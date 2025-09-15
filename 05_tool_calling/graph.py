import os
import requests
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from datetime import datetime
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- Configuration ---
load_dotenv(dotenv_path="studio/.env")

# Se añade la comprobación para la nueva API Key del tiempo
if not all([os.getenv("OPENAI_API_KEY"), os.getenv("TAVILY_API_KEY"), os.getenv("WEATHER_API_KEY")]):
   raise ValueError("One or more required API keys (OPENAI, TAVILY, WEATHER) are not set in studio/.env")

# --- 1. Define Tools ---

tavily_tool = TavilySearchResults(max_results=3)

@tool
def get_current_weather(city: str) -> str:
    """
    Gets the current weather conditions for a specified city.
    Use this tool to find out the temperature and weather description (e.g., sunny, rainy).
    """
    print(f"--- Calling Tool: get_current_weather for '{city}' ---")
    api_key = os.getenv("WEATHER_API_KEY")
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        location = data["location"]["name"]
        temp_c = data["current"]["temp_c"]
        condition = data["current"]["condition"]["text"]
        
        return f"In {location}, the current temperature is {temp_c}°C with {condition}."
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            return f"Error: City '{city}' not found."
        else:
            return f"HTTP error occurred: {http_err}"
    except Exception as e:
        return f"An error occurred: {e}"

# La lista de herramientas ahora incluye el buscador y la nueva herramienta del tiempo
tools = [tavily_tool, get_current_weather]

# --- 2. Define the State ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- 3. Define the System Prompt and Nodes ---

# Nuevo System Prompt para el rol de "Asistente de Planes y Ocio"
AGENT_SYSTEM_PROMPT = """
You are a helpful and proactive "Leisure and Planning Assistant". Your goal is to provide the current weather for a city and then suggest a fun, suitable activity based on that weather.

**Your Thought Process (ReAct):**
1.  **Analyze**: The user asks for the weather in a city (e.g., "What's the weather in Paris?").
2.  **Crucial First Step**: You **MUST** first call the `get_current_weather` tool for that city.
3.  **Observe**: Analyze the weather tool's output (e.g., "In Paris, the current temperature is 12°C with Light rain.").
4.  **Reason and Act (Second Step)**: Based on the weather, reason about a suitable activity.
    - If it's raining, search for indoor activities like "famous museums in Paris" using the `tavily_search` tool.
    - If it's sunny, search for outdoor activities like "best parks in Paris" using the `tavily_search` tool.
5.  **Synthesize and Respond**: Combine the weather information and the activity suggestion into a single, helpful response. For example: "The weather in Paris is currently 12°C with light rain. It's a perfect day to visit the Louvre Museum!"
"""
current_date = datetime.now().strftime("%Y-%m-%d")
SYSTEM_MESSAGE = SystemMessage(content=AGENT_SYSTEM_PROMPT.format(current_date=current_date))

llm = ChatOpenAI(model="gpt-4o")
model_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    """Invokes the LLM with the system prompt and the current conversation state."""
    print("--- Calling Agent Node ---")
    context = [SYSTEM_MESSAGE] + state['messages']
    response = model_with_tools.invoke(context)
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
