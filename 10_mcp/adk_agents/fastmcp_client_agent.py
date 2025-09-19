# This script defines a Google ADK agent that acts as a client
# to the FastMCP server defined in `10_fastmcp_server.py`.
#
# Prerequisites:
# 1. Google ADK installed (`pip install google-adk`).
# 2. The `10_fastmcp_server.py` must be running in a separate terminal.

import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, HttpServerParameters

# Define the FastMCP server's address.
# This must match the host and port where the server is running.
FASTMCP_SERVER_URL = "http://localhost:8000"

root_agent = LlmAgent(
   model='gemini-2.0-flash',
   name='fastmcp_greeter_agent',
   instruction='You are a friendly assistant that can greet people by their name. Use the "greet" tool.',
   tools=[
       MCPToolset(
           connection_params=HttpServerParameters(
               url=FASTMCP_SERVER_URL,
           ),
           # Optional: Filter which tools from the MCP server are exposed.
           # For this example, we only need the 'greet' tool.
           tool_filter=['greet']
       )
   ],
)
