# 05. Tool Use (Function Calling)

## What is Tool Use?

The **Tool Use** pattern, often implemented via **Function Calling**, enables an agent to interact with the outside world. It allows the core LLM to go beyond its internal knowledge by calling external APIs, databases, services, or even executing code to fulfill a user's request.

The process involves:
1.  **Tool Definition**: Describing external functions (their purpose, name, and parameters) to the LLM.
2.  **LLM Decision**: The LLM analyzes a user's request and decides if a tool is needed.
3.  **Function Call Generation**: If a tool is needed, the LLM generates a structured output (like JSON) specifying the tool's name and the arguments to use.
4.  **Tool Execution**: The agent's framework executes the actual function with the provided arguments.
5.  **Result Processing**: The result from the tool is sent back to the LLM, which then formulates a final response to the user.

## Why is it useful?

LLMs are fundamentally disconnected from the real world; their knowledge is static and limited to their training data. Tool Use breaks these limitations, allowing agents to:
-   Access real-time, up-to-date information (e.g., weather, stock prices).
-   Interact with private or user-specific data (e.g., company databases, personal calendars).
-   Perform precise calculations or execute complex code.
-   Trigger real-world actions (e.g., sending an email, controlling smart devices).

## Key Points

-   **External Interaction**: Tool Use is the bridge between an LLM's reasoning and external systems.
-   **Structured Output**: The LLM doesn't execute code directly; it generates a request for a tool to be executed by the agent framework.
-   **Clarity is Crucial**: The descriptions of the tools must be very clear for the LLM to understand when and how to use them correctly.
-   **Framework Support**: Frameworks like LangChain, CrewAI, and Google ADK provide robust support for defining and integrating tools into agent workflows.

## Practical Use Cases

-   **Information Retrieval**: A weather agent using a weather API.
-   **Database Interaction**: An e-commerce agent checking product inventory via an internal API.
-   **Communications**: A personal assistant agent sending an email using an email API.
-   **Code Execution**: A coding assistant using a code interpreter to run and analyze a snippet of code.
