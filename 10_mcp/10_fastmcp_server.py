# This script demonstrates how to create a simple MCP server using FastMCP.
# It exposes a single tool that generates a greeting.
#
# To run this server:
# 1. Install FastMCP:
#    pip install fastmcp
# 2. Run the script from your terminal:
#    python 10_fastmcp_server.py

from fastmcp import FastMCP

# Initialize the FastMCP server.
mcp_server = FastMCP()

# The `@mcp_server.tool` decorator registers this Python function as an MCP tool.
# The docstring and type hints are automatically used to create the tool's
# schema, which tells the LLM how to use it.
@mcp_server.tool
def greet(name: str) -> str:
    """
    Generates a personalized greeting.

    Args:
        name: The name of the person to greet.

    Returns:
        A greeting string.
    """
    return f"Hello, {name}! Nice to meet you."

# This block allows the server to be run directly from the command line.
if __name__ == "__main__":
    print("Starting FastMCP server on http://127.0.0.1:8000")
    print("Press Ctrl+C to stop.")
    mcp_server.run(
        transport="http",
        host="127.0.0.1",
        port=8000
    )
