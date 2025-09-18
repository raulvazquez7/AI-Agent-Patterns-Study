import os
import functools
from dotenv import load_dotenv
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

# LangChain & LangGraph Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, RemoveMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
# NUEVO: Importar RunnableConfig para el tipado correcto
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Project-specific imports
from supabase import create_client, Client as SupabaseClient
from src.database import add_turn_to_vector_memory, search_vector_memory

# --- 1. Cargar configuración y clientes ---
load_dotenv(dotenv_path="studio/.env")
if not all([os.getenv("OPENAI_API_KEY"), os.getenv("TAVILY_API_KEY")]):
   raise ValueError("API keys for OPENAI and TAVILY must be set in studio/.env")

# --- 1.1. Configuración de Modelos y Clientes ---
llm = ChatOpenAI(model="gpt-4o")
summarization_llm = ChatOpenAI(model="gpt-4-turbo")
entity_extraction_llm = ChatOpenAI(model="gpt-4-turbo")

# Simplificado: Usar siempre el modelo de embeddings de OpenAI que es más estable en la inicialización.
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
embeddings_model = OpenAIEmbeddings(model=EMBEDDINGS_MODEL_NAME)

# Cliente de Supabase
supabase_client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))

# --- 2. Definir el ESTADO del Grafo (con Ficha de Usuario) ---
SUMMARIZATION_TURN_THRESHOLD = 3

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    turn_count: int
    user_profile: dict
    next: Literal["Researcher", "Writer", "Supervisor", "Conversational", END]

# --- 3. Definir herramientas ---
tavily_tool = TavilySearch(max_results=3)

# --- 4. Definir Nodos ---

# --- 4.1. Prompts ---
# (Los prompts largos se mantienen igual)
SUPERVISOR_SYSTEM_PROMPT = """You are a project manager supervising a team of three specialists: a Researcher, a Writer, and a Conversational Agent.
Your goal is to orchestrate the team to answer the user's request completely.

Team Members:
- Researcher: An expert in using web search tools to find factual, up-to-date information.
- Writer: A skilled technical writer who synthesizes the gathered information into a final, coherent report.
- Conversational Agent: A friendly assistant who handles greetings, farewells, and general chit-chat.

Delegation Logic:
1.  **Analyze the request:** Based on the current conversation, decide which specialist should act next.
2.  **To Conversational Agent:** If the user is just making small talk, greeting, or saying goodbye, delegate to the Conversational Agent.
3.  **To Researcher:** If the request requires new information or fact-checking, delegate to the Researcher.
4.  **To Writer:** Once the research is complete (indicated by a ToolMessage in the history), delegate to the Writer to compose the final answer.
5.  **FINISH:** Once the Writer has provided the complete report, or the conversation is over, the task is done. Respond with FINISH.

Constraints:
- You must not perform any work yourself. Your sole function is to delegate to the appropriate specialist.
"""

RESEARCHER_SYSTEM_PROMPT = """You are a senior research analyst. Your primary role is to find relevant, factual, and up-to-date information on a given topic.
Your Process:
1.  Analyze the user's request and the conversation history to understand the information needed.
2.  Formulate a precise and effective search query for the `tavily_search` tool.
3.  Execute the tool call.
Constraints:
- You **must** use the `tavily_search` tool to gather information.
- Your output should be the tool call itself. Do **not** synthesize or write the final answer to the user.
"""

WRITER_SYSTEM_PROMPT = """You are a professional technical writer. Your responsibility is to synthesize the information provided by the Researcher into a clear, concise, and well-structured report.
Your Process:
1.  Carefully review the entire conversation history, paying close attention to the `ToolMessage` which contains the research findings.
2.  Compose a comprehensive final answer to the user's original request based **exclusively** on the provided information.
Constraints:
- Do **not** use any tools or invent information. Your output is the final, user-facing response.
"""

CONVERSATIONAL_SYSTEM_PROMPT = """You are a friendly and professional conversational assistant.
Your role is to handle greetings, farewells, and general chit-chat.
Keep your responses brief, polite, and helpful. Do not answer research questions or write reports.
"""

# --- 4.2. "Fábrica" de Agentes y Cadenas ---
def create_agent(system_prompt: str, tools: list = None):
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("placeholder", "{messages}")])
    return prompt_template | llm.bind_tools(tools) if tools else prompt_template | llm

research_agent_chain = create_agent(RESEARCHER_SYSTEM_PROMPT, tools=[tavily_tool])
writer_agent_chain = create_agent(WRITER_SYSTEM_PROMPT)
conversational_agent_chain = create_agent(CONVERSATIONAL_SYSTEM_PROMPT)

# --- 4.3. Nodos de Agentes (Ahora con Memoria y Perfil) ---

# Este "envoltorio" es para los agentes complejos que SÍ necesitan la memoria a largo plazo.
# CORREGIDO: El parámetro config ahora está tipado como RunnableConfig
def agent_node_wrapper(state: AgentState, agent_chain, agent_name: str, config: RunnableConfig):
    print(f"--- {agent_name.upper()} ---")
    
    # 1. Recuperar memoria vectorial relevante
    last_user_message = state["messages"][-1].content
    retrieved_turns = search_vector_memory(
        supabase_client, config["configurable"]["thread_id"], last_user_message, embeddings_model
    )
    
    # 2. Construir el contexto enriquecido
    context = []
    if state.get("summary"):
        context.append(SystemMessage(f"Summary of the conversation so far:\n{state['summary']}"))
    if state.get("user_profile"):
        context.append(SystemMessage(f"Known user profile:\n{state['user_profile']}"))
    if retrieved_turns:
        context.append(SystemMessage(f"Relevant past turns:\n" + "\n".join(retrieved_turns)))
    
    context.extend(state["messages"])
    
    # 3. Invocar al agente
    result = agent_chain.invoke({"messages": context})
    return {"messages": [result]}

researcher_node = functools.partial(agent_node_wrapper, agent_chain=research_agent_chain, agent_name="Researcher")
writer_node = functools.partial(agent_node_wrapper, agent_chain=writer_agent_chain, agent_name="Writer")

# NUEVO: Nodo dedicado y ligero para el agente conversacional.
def conversational_node(state: AgentState):
    print("--- CONVERSATIONAL AGENT ---")
    
    # Construye un contexto simple, solo con el perfil de usuario para personalizar el saludo.
    context = []
    if name := state.get("user_profile", {}).get("name"):
        context.append(SystemMessage(f"The user's name is {name}. Greet them by name."))
    
    context.extend(state["messages"])
    
    result = conversational_agent_chain.invoke({"messages": context})
    return {"messages": [result]}

tool_node = ToolNode([tavily_tool])

# --- 4.4. Nodo Supervisor ---
class SupervisorRouter(BaseModel):
    next: Literal["Researcher", "Writer", "Conversational", "FINISH"]

supervisor_chain = (
    ChatPromptTemplate.from_messages([("system", SUPERVISOR_SYSTEM_PROMPT), ("placeholder", "{messages}")])
    | llm.with_structured_output(SupervisorRouter)
)

def supervisor_node(state: AgentState):
    print("--- SUPERVISOR ---")
    response = supervisor_chain.invoke({"messages": state["messages"]})
    print(f"Supervisor decision: '{response.next}'")
    if response.next == "FINISH":
        return {"next": "end_of_turn"}
    return {"next": response.next}

# --- 4.5. Nodos de Mantenimiento de Memoria (NUEVOS) ---
class UserProfile(BaseModel):
    name: Optional[str] = Field(description="The user's name.")
    location: Optional[str] = Field(description="The user's city or location.")
    preferences: Optional[list[str]] = Field(description="A list of the user's explicit preferences.")

def entity_extraction_node(state: AgentState):
    print("--- MEMORY: EXTRACTING ENTITIES ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an entity extraction expert. From the last user message, extract the user's name, location, or any stated preferences. If no new information is present, call the tool with empty or null values."),
        ("user", "{last_message}")
    ])
    extraction_chain = prompt | entity_extraction_llm.with_structured_output(UserProfile)
    
    try:
        extracted_data = extraction_chain.invoke({"last_message": state["messages"][-1].content})
        
        # Actualizar el perfil de usuario sin sobrescribir datos existentes
        current_profile = state.get("user_profile", {})
        for key, value in extracted_data.dict().items():
            if value: # Solo actualizar si se extrajo un nuevo valor
                current_profile[key] = value
        
        print(f"--- Updated user profile: {current_profile} ---")
        return {"user_profile": current_profile}
    except Exception:
        # Si la extracción falla, simplemente continuamos
        return {}

# CORREGIDO: El parámetro config ahora está tipado como RunnableConfig
def update_vector_memory_node(state: AgentState, config: RunnableConfig):
    print("--- MEMORY: UPDATING VECTOR STORE ---")
    # Guardar el último turno (pregunta del usuario y respuesta del bot)
    last_user_message = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    last_ai_message = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    
    if last_user_message and last_ai_message:
        turn_content = f"User: {last_user_message.content}\nAI: {last_ai_message.content}"
        add_turn_to_vector_memory(
            supabase_client, config["configurable"]["thread_id"], turn_content, embeddings_model
        )
    return {}
    
def end_of_turn_node(state: AgentState):
    print("--- END OF TURN ---")
    return {"turn_count": state.get("turn_count", 0) + 1}

def summarizer_node(state: AgentState):
    print("--- SUMMARIZING & PRUNING ---")
    conversation = state['messages']
    summary_prompt = [
        SystemMessage(content="Summarize the conversation..."),
        HumanMessage(content=f"CURRENT SUMMARY:\n{state.get('summary', 'No summary yet.')}\n\nNEW MESSAGES:\n" + "\n".join(f"- {type(m).__name__}: {m.content}" for m in conversation))
    ]
    new_summary = summarization_llm.invoke(summary_prompt).content
    messages_to_remove = [RemoveMessage(id=m.id) for m in state["messages"]]
    print(f"--- New summary generated. Message window reset. ---")
    return {"summary": new_summary, "messages": messages_to_remove, "turn_count": 0}

# --- 5. Construir el grafo (con el nuevo flujo de memoria) ---
workflow = StateGraph(AgentState)

# Añadir todos los nodos
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Writer", writer_node)
workflow.add_node("Conversational", conversational_node)
workflow.add_node("call_tool", tool_node)
workflow.add_node("end_of_turn", end_of_turn_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("update_vector_memory", update_vector_memory_node)
workflow.add_node("summarizer", summarizer_node)
workflow.set_entry_point("Supervisor")

# --- Definir Conexiones del Grafo ---
def after_researcher_action(state: AgentState):
    return "call_tool" if state["messages"][-1].tool_calls else "Supervisor"

def should_summarize(state: AgentState):
    return "summarizer" if state.get("turn_count", 0) >= SUMMARIZATION_TURN_THRESHOLD else END

# Flujo principal
workflow.add_conditional_edges(
    "Supervisor", lambda state: state["next"],
    {"Researcher": "Researcher", "Writer": "Writer", "Conversational": "Conversational", "end_of_turn": "end_of_turn"}
)
workflow.add_conditional_edges(
    "Researcher", after_researcher_action, {"call_tool": "call_tool", "Supervisor": "Supervisor"}
)

# Flujo hacia el final del turno (donde el usuario ya tiene su respuesta)
workflow.add_edge("Writer", "end_of_turn")
workflow.add_edge("Conversational", "end_of_turn")
workflow.add_edge("call_tool", "Researcher")

# Flujo de mantenimiento de memoria (la parte "asíncrona")
workflow.add_edge("end_of_turn", "entity_extraction")
workflow.add_edge("entity_extraction", "update_vector_memory")
workflow.add_conditional_edges("update_vector_memory", should_summarize, {"summarizer": "summarizer", END: END})
workflow.add_edge("summarizer", END)

# --- 6. Compilar el grafo ---
app = workflow.compile()
console_app = workflow.compile(checkpointer=MemorySaver())
