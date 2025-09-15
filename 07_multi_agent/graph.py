import os
from dotenv import load_dotenv
from typing import Annotated, Literal
# Corregido: Usar typing_extensions para mayor compatibilidad con Pydantic v2
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
# Corregido: Usar el nuevo paquete para Tavily
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
# Corregido: Usar pydantic v2 directamente
from pydantic import BaseModel

# --- 1. Cargar configuración y variables de entorno ---
load_dotenv(dotenv_path="studio/.env")
if not all([os.getenv("OPENAI_API_KEY"), os.getenv("TAVILY_API_KEY")]):
   raise ValueError("API keys for OPENAI and TAVILY must be set in studio/.env")

# --- 2. Definir el estado del grafo (la memoria compartida del equipo) ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: Literal["Researcher", "Writer", "Supervisor", END]

# --- 3. Definir las herramientas que usarán nuestros agentes ---
tavily_tool = TavilySearch(max_results=3)

# --- 4. Definir los agentes especialistas y el supervisor ---
llm = ChatOpenAI(model="gpt-4o")

def create_agent(system_prompt: str, tools: list = None):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{messages}"),
    ])
    if tools:
        model_with_tools = llm.bind_tools(tools)
        return prompt_template | model_with_tools
    else:
        return prompt_template | llm

# Cadenas de agentes
research_agent_chain = create_agent(
    system_prompt="You are a senior research analyst. Your role is to find relevant information on a given topic using the available search tool.",
    tools=[tavily_tool]
)
writer_agent_chain = create_agent(
    system_prompt="You are a professional technical writer. Your task is to take the research provided and compose a clear, concise, and well-structured report."
)

# Solución: Envolver los agentes en nodos explícitos
def researcher_node(state: AgentState):
    print("--- RESEARCHER ---")
    result = research_agent_chain.invoke({"messages": state["messages"]})
    return {"messages": [result]}

def writer_node(state: AgentState):
    print("--- WRITER ---")
    result = writer_agent_chain.invoke({"messages": state["messages"]})
    return {"messages": [result]}

tool_node = ToolNode([tavily_tool])

# Supervisor
class SupervisorRouter(BaseModel):
    next: Literal["Researcher", "Writer", "FINISH"]

supervisor_chain = (
    ChatPromptTemplate.from_messages([
        ("system", 
         "You are a supervisor managing a team of two specialists: a Researcher and a Writer. "
         "Based on the user's request, decide which specialist should act next. "
         "If the research is done, delegate to the Writer. "
         "If the report is written, or if the initial request is simple, you can finish. "
         "Do not do any work yourself, only delegate."),
        ("placeholder", "{messages}"),
    ])
    | llm.with_structured_output(SupervisorRouter)
)

def supervisor_node(state: AgentState):
    print("--- SUPERVISOR ---")
    response = supervisor_chain.invoke({"messages": state["messages"]})
    print(f"Supervisor decision: '{response.next}'")
    if response.next == "FINISH":
        return {"next": END}
    return {"next": response.next}

# --- 5. Construir el grafo y definir las conexiones ---
workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Writer", writer_node)
workflow.add_node("call_tool", tool_node)

workflow.set_entry_point("Supervisor")

workflow.add_conditional_edges(
    "Supervisor",
    lambda state: state["next"],
    {"Researcher": "Researcher", "Writer": "Writer", END: END},
)

def after_researcher_action(state: AgentState):
    # Ahora esto funciona porque el último mensaje es garantizado un AIMessage del Researcher
    if state["messages"][-1].tool_calls:
        return "call_tool"
    return "Supervisor"

workflow.add_conditional_edges(
    "Researcher",
    after_researcher_action,
    {"call_tool": "call_tool", "Supervisor": "Supervisor"},
)

workflow.add_edge("Writer", "Supervisor")
# Corregido: El resultado de la herramienta vuelve al Investigador para que lo procese
workflow.add_edge("call_tool", "Researcher")

app = workflow.compile()
print("Multi-agent graph compiled successfully!")
