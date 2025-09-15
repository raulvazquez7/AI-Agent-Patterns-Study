# This file contains multiple examples from the "Google ADK" framework,
# demonstrating different multi-agent collaboration structures.

# ==============================================================================
# Example 1: Hierarchical Agent Structure
# ==============================================================================
print("--- Example 1: Hierarchical ---")
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator

# Correctly implement a custom agent by extending BaseAgent
class TaskExecutor(BaseAgent):
   """A specialized agent with custom, non-LLM behavior."""
   name: str = "TaskExecutor"
   description: str = "Executes a predefined task."

   async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
       """Custom implementation logic for the task."""
       yield Event(author=self.name, content="Task finished successfully.")

# Define individual agents with proper initialization
greeter = LlmAgent(
   name="Greeter",
   model="gemini-2.0-flash-exp",
   instruction="You are a friendly greeter."
)
task_doer = TaskExecutor()

# Create a parent agent and assign its sub-agents
coordinator = LlmAgent(
   name="Coordinator",
   model="gemini-2.0-flash-exp",
   description="A coordinator that can greet users and execute tasks.",
   instruction="When asked to greet, delegate to the Greeter. When asked to perform a task, delegate to the TaskExecutor.",
   sub_agents=[
       greeter,
       task_doer
   ]
)

assert greeter.parent_agent == coordinator
assert task_doer.parent_agent == coordinator
print("Agent hierarchy created successfully.\n")


# ==============================================================================
# Example 2: Loop Agent for Iterative Workflows
# ==============================================================================
print("--- Example 2: Loop Agent ---")
from google.adk.agents import LoopAgent, EventActions

class ConditionChecker(BaseAgent):
   """A custom agent that checks for a 'completed' status."""
   name: str = "ConditionChecker"
   description: str = "Checks if a process is complete and signals the loop to stop."

   async def _run_async_impl(
       self, context: InvocationContext
   ) -> AsyncGenerator[Event, None]:
       status = context.session.state.get("status", "pending")
       if (status == "completed"):
           yield Event(author=self.name, actions=EventActions(escalate=True))
       else:
           yield Event(author=self.name, content="Condition not met, continuing loop.")

process_step = LlmAgent(
   name="ProcessingStep",
   model="gemini-2.0-flash-exp",
   instruction="Perform your task. If you are the final step, update session state by setting 'status' to 'completed'."
)

poller = LoopAgent(
   name="StatusPoller",
   max_iterations=10,
   sub_agents=[
       process_step,
       ConditionChecker()
   ]
)
print("LoopAgent 'poller' created.\n")


# ==============================================================================
# Example 3: Sequential Agent for Linear Workflows
# ==============================================================================
print("--- Example 3: Sequential Agent ---")
from google.adk.agents import SequentialAgent, Agent

step1 = Agent(name="Step1_Fetch", output_key="data")
step2 = Agent(
   name="Step2_Process",
   instruction="Analyze the information found in state['data'] and provide a summary."
)

pipeline = SequentialAgent(
   name="MyPipeline",
   sub_agents=[step1, step2]
)
print("SequentialAgent 'pipeline' created.\n")


# ==============================================================================
# Example 4: Parallel Agent for Concurrent Execution
# ==============================================================================
print("--- Example 4: Parallel Agent ---")
from google.adk.agents import ParallelAgent

weather_fetcher = Agent(
   name="weather_fetcher",
   model="gemini-2.0-flash-exp",
   instruction="Fetch the weather for the given location and return only the weather report.",
   output_key="weather_data"
)

news_fetcher = Agent(
   name="news_fetcher",
   model="gemini-2.0-flash-exp",
   instruction="Fetch the top news story for the given topic and return only that story.",
   output_key="news_data"
)

data_gatherer = ParallelAgent(
   name="data_gatherer",
   sub_agents=[
       weather_fetcher,
       news_fetcher
   ]
)
print("ParallelAgent 'data_gatherer' created.\n")


# ==============================================================================
# Example 5: Agent as a Tool
# ==============================================================================
print("--- Example 5: Agent as a Tool ---")
from google.adk.tools import agent_tool

def generate_image(prompt: str) -> dict:
   """Simulates generating an image based on a prompt."""
   print(f"TOOL: Generating image for prompt: '{prompt}'")
   return {
       "status": "success",
       "image_bytes": b"mock_image_data",
       "mime_type": "image/png"
   }

image_generator_agent = LlmAgent(
   name="ImageGen",
   model="gemini-2.0-flash",
   description="Generates an image based on a detailed text prompt.",
   instruction=(
       "Take the user's request and use the `generate_image` tool to create the image. "
       "The user's entire request should be used as the 'prompt' argument. "
       "After the tool returns the image bytes, you MUST output the image."
   ),
   tools=[generate_image]
)

image_tool = agent_tool.AgentTool(
   agent=image_generator_agent,
   description="Use this tool to generate an image. The input should be a descriptive prompt."
)

artist_agent = LlmAgent(
   name="Artist",
   model="gemini-2.0-flash",
   instruction=(
       "You are a creative artist. First, invent a creative prompt for an image. "
       "Then, use the `ImageGen` tool to generate the image using your prompt."
   ),
   tools=[image_tool]
)
print("Hierarchy with 'Agent as a Tool' created successfully.")
