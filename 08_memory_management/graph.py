import os
from dotenv import load_dotenv
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, RemoveMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

# --- 1. Cargar configuración ---
load_dotenv(dotenv_path="studio/.env")
if not all([os.getenv("OPENAI_API_KEY"), os.getenv("TAVILY_API_KEY")]):
   raise ValueError("API keys for OPENAI and TAVILY must be set in studio/.env")

# --- 2. Definir el ESTADO con la nueva ruta "Conversational" ---
SUMMARIZATION_TURN_THRESHOLD = 3

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    turn_count: int
    next: Literal["Researcher", "Writer", "Supervisor", "Conversational", END]

# --- 3. Definir herramientas ---
tavily_tool = TavilySearch(max_results=3)

# --- 4. Definir agentes, supervisor y nodos de memoria ---
llm = ChatOpenAI(model="gpt-4o")
summarization_llm = ChatOpenAI(model="gpt-4-turbo")

# --- 4.1. Definir los System Prompts para cada rol ---
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

def create_agent(system_prompt: str, tools: list = None):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt), ("placeholder", "{messages}"),
    ])
    if tools:
        return prompt_template | llm.bind_tools(tools)
    return prompt_template | llm

# --- Nodos de Agentes Especialistas ---
research_agent_chain = create_agent(RESEARCHER_SYSTEM_PROMPT, tools=[tavily_tool])
writer_agent_chain = create_agent(WRITER_SYSTEM_PROMPT)
conversational_agent_chain = create_agent(CONVERSATIONAL_SYSTEM_PROMPT)

def researcher_node(state: AgentState):
    print("--- RESEARCHER ---")
    context = list(state["messages"])
    if state.get("summary"):
        context.insert(0, SystemMessage(f"Summary of the conversation so far:\n{state['summary']}"))
    result = research_agent_chain.invoke({"messages": context})
    return {"messages": [result]}

def writer_node(state: AgentState):
    print("--- WRITER ---")
    context = list(state["messages"])
    if state.get("summary"):
        context.insert(0, SystemMessage(f"Summary of the conversation so far:\n{state['summary']}"))
    result = writer_agent_chain.invoke({"messages": context})
    return {"messages": [result]}

def conversational_node(state: AgentState):
    print("--- CONVERSATIONAL AGENT ---")
    result = conversational_agent_chain.invoke({"messages": state["messages"]})
    return {"messages": [result]}

tool_node = ToolNode([tavily_tool])

# --- Nodo Supervisor ---
class SupervisorRouter(BaseModel):
    next: Literal["Researcher", "Writer", "Conversational", "FINISH"]

supervisor_chain = (
    ChatPromptTemplate.from_messages([
        ("system", SUPERVISOR_SYSTEM_PROMPT), ("placeholder", "{messages}"),
    ]) | llm.with_structured_output(SupervisorRouter)
)

def supervisor_node(state: AgentState):
    print("--- SUPERVISOR ---")
    response = supervisor_chain.invoke({"messages": state["messages"]})
    print(f"Supervisor decision: '{response.next}'")
    if response.next == "FINISH":
        return {"next": "end_of_turn"}
    return {"next": response.next}

# --- Nodos de Gestión de Memoria ---
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

# --- 5. Construir el grafo ---
workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Writer", writer_node)
workflow.add_node("Conversational", conversational_node)
workflow.add_node("call_tool", tool_node)
workflow.add_node("end_of_turn", end_of_turn_node)
workflow.add_node("summarizer", summarizer_node)
workflow.set_entry_point("Supervisor")

# --- Definir Conexiones del Grafo ---
def after_researcher_action(state: AgentState):
    return "call_tool" if state["messages"][-1].tool_calls else "Supervisor"

def should_summarize(state: AgentState):
    return "summarizer" if state.get("turn_count", 0) >= SUMMARIZATION_TURN_THRESHOLD else END

workflow.add_conditional_edges(
    "Supervisor",
    lambda state: state["next"],
    {"Researcher": "Researcher", "Writer": "Writer", "Conversational": "Conversational", "end_of_turn": "end_of_turn"},
)
workflow.add_conditional_edges(
    "Researcher", after_researcher_action, {"call_tool": "call_tool", "Supervisor": "Supervisor"}
)
workflow.add_conditional_edges("end_of_turn", should_summarize, {"summarizer": "summarizer", END: END})

workflow.add_edge("Writer", "end_of_turn")
workflow.add_edge("call_tool", "Researcher")
workflow.add_edge("Conversational", "end_of_turn")
workflow.add_edge("summarizer", END)

# --- 6. Compilar el grafo ---
app = workflow.compile()
print("Graph compiled for LangGraph Studio (no persistence).")

memory = MemorySaver()
console_app = workflow.compile(checkpointer=memory)
print("Graph re-compiled with in-memory checkpointer for console.")

# --- Bloque de ejecución para consola (opcional) ---
if __name__ == '__main__':
    import uuid
    thread_id = str(uuid.uuid4())
    run_config = {"configurable": {"thread_id": thread_id}}
    print(f"--- Starting new conversation with Thread ID: {thread_id} ---")
    
    # Ejemplo de uso en consola
    # console_app.invoke({"messages": [HumanMessage(content="What are the latest AI trends?")]}, config=run_config)
    # console_app.invoke({"messages": [HumanMessage(content="Write a blog post about it.")]}, config=run_config)
