# This script defines a Google ADK agent that can interact with the local filesystem.
# It uses an MCP server distributed as a Node.js package.
#
# Prerequisites:
# 1. Google ADK installed (`pip install google-adk`).
# 2. Node.js and npx installed.

import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# Create a reliable absolute path to a folder named 'mcp_managed_files'
# within the same directory as this agent script.
# This ensures the agent works out-of-the-box for demonstration.
TARGET_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_managed_files")

# Ensure the target directory exists before the agent needs it.
os.makedirs(TARGET_FOLDER_PATH, exist_ok=True)
print(f"MCP Filesystem Agent will operate in: {TARGET_FOLDER_PATH}")

root_agent = LlmAgent(
   model='gemini-2.0-flash',
   name='filesystem_assistant_agent',
   instruction=(
       'Help the user manage their files. You can list files, read files, and write files. '
       f'You are operating in the following directory: {TARGET_FOLDER_PATH}'
   ),
   tools=[
       MCPToolset(
           connection_params=StdioServerParameters(
               # npx will automatically download and run the MCP filesystem server.
               command='npx',
               args=[
                   "-y",  # Argument for npx to auto-confirm install
                   "@modelcontextprotocol/server-filesystem",
                   # This MUST be an absolute path to the folder the agent can access.
                   TARGET_FOLDER_PATH,
               ],
           ),
           # Optional: You can filter which tools from the MCP server are exposed.
           # For example, to only allow reading:
           # tool_filter=['list_directory', 'read_file']
       )
   ],
)

