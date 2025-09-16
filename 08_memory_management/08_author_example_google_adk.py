"""
This script contains a collection of code examples demonstrating memory management
features within the Google Agent Developer Kit (ADK), as described in Chapter 8.
"""
import time
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner
from google.genai.types import Content, Part
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.invocation_context import InvocationContext

# --- Example 1: SessionService Implementations ---
# The SessionService manages chat threads, including their history (events)
# and temporary data (state).

print("--- Example 1: SessionService Options ---")

# a) In-memory session service (for testing, not persistent)
print("Initializing InMemorySessionService...")
in_memory_session_service = InMemorySessionService()
print("-> Done.")

# b) Database session service (for persistence)
# Requires additional setup: pip install google-adk[sqlalchemy] and a DB driver.
# from google.adk.sessions import DatabaseSessionService
# db_url = "sqlite:///./my_agent_data.db"
# print(f"\nInitializing DatabaseSessionService with URL: {db_url}...")
# database_session_service = DatabaseSessionService(db_url=db_url)
# print("-> Done (conceptual).")

# c) Vertex AI session service (for scalable production on GCP)
# Requires GCP setup: pip install google-adk[vertexai]
# from google.adk.sessions import VertexAiSessionService
# PROJECT_ID = "your-gcp-project-id"
# LOCATION = "us-central1"
# print("\nInitializing VertexAiSessionService...")
# vertex_session_service = VertexAiSessionService(project=PROJECT_ID, location=LOCATION)
# print("-> Done (conceptual).")

print("-" * 20, "\n")


# --- Example 2: State Update via output_key ---
# The simplest way to save an agent's text response to the session state.

print("--- Example 2: State Update with output_key ---")

greeting_agent = LlmAgent(
   name="Greeter",
   model="gemini-2.0-flash",
   instruction="Generate a short, friendly greeting.",
   output_key="last_greeting"  # The agent's response will be saved here
)

app_name, user_id, session_id = "state_app", "user1", "session1"
session_service = InMemorySessionService()
runner = Runner(
   agent=greeting_agent,
   app_name=app_name,
   session_service=session_service
)
session = session_service.create_session(
   app_name=app_name, user_id=user_id, session_id=session_id
)

print(f"Initial state: {session.state}")

user_message = Content(parts=[Part(text="Hello")])
print("\n--- Running the agent ---")
for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_message):
   if event.is_final_response():
     print("Agent responded.")

updated_session = session_service.get_session(app_name, user_id, session_id)
print(f"State after agent run: {updated_session.state}")

print("-" * 20, "\n")


# --- Example 3: State Update via Tool ---
# The recommended approach for more complex state updates.

print("--- Example 3: State Update with a Tool ---")

def log_user_login(tool_context: ToolContext) -> dict:
   """Updates session state upon user login, encapsulating all related logic."""
   state = tool_context.state
   login_count = state.get("user:login_count", 0) + 1
   state["user:login_count"] = login_count
   state["task_status"] = "active"
   state["user:last_login_ts"] = time.time()
   state["temp:validation_needed"] = True
   print("State updated from within the `log_user_login` tool.")
   return {"status": "success", "message": f"User login tracked. Total logins: {login_count}."}

# Simulate a tool call (in a real app, the Runner would handle this)
session_service_tool = InMemorySessionService()
app_name_tool, user_id_tool, session_id_tool = "state_app_tool", "user3", "session3"
session_tool = session_service_tool.create_session(
   app_name=app_name_tool,
   user_id=user_id_tool,
   session_id=session_id_tool,
   state={"user:login_count": 0, "task_status": "idle"}
)
print(f"Initial state: {session_tool.state}")

mock_context = ToolContext(
   invocation_context=InvocationContext(
       app_name=app_name_tool, user_id=user_id_tool, session_id=session_id_tool,
       session=session_tool, session_service=session_service_tool
   )
)

log_user_login(mock_context)

updated_session_tool = session_service_tool.get_session(app_name_tool, user_id_tool, session_id_tool)
print(f"State after tool execution: {updated_session_tool.state}")

print("-" * 20, "\n")


# --- Example 4: MemoryService for Long-Term Knowledge ---
# The MemoryService manages a persistent, searchable repository of information.

print("--- Example 4: MemoryService Options ---")

# a) In-memory memory service (for testing, not persistent)
from google.adk.memory import InMemoryMemoryService
print("Initializing InMemoryMemoryService...")
in_memory_memory_service = InMemoryMemoryService()
print("-> Done.")

# b) Vertex AI RAG memory service (for scalable production on GCP)
# Requires GCP setup: pip install google-adk[vertexai]
# from google.adk.memory import VertexAiRagMemoryService
# RAG_CORPUS_RESOURCE_NAME = "projects/your-gcp-project-id/locations/us-central1/ragCorpora/your-corpus-id"
# print("\nInitializing VertexAiRagMemoryService...")
# vertex_rag_service = VertexAiRagMemoryService(rag_corpus=RAG_CORPUS_RESOURCE_NAME)
# print("-> Done (conceptual).")

# c) Vertex AI Memory Bank service (managed service for facts and preferences)
# from google.adk.memory import VertexAiMemoryBankService
# print("\nInitializing VertexAiMemoryBankService...")
# memory_bank_service = VertexAiMemoryBankService(
#    project="PROJECT_ID",
#    location="LOCATION",
#    agent_engine_id="your-engine-id"
# )
# print("-> Done (conceptual).")


