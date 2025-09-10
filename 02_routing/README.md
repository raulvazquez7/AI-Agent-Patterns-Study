# 02. Routing

## What is Routing?

The **Routing** pattern introduces conditional logic into an agent's workflow. Unlike a simple sequential path (like in Prompt Chaining), a routing agent can analyze an input and dynamically decide which tool, function, or sub-agent to use next. This allows for more flexible, adaptive, and context-aware behavior.

It enables a system to move from a fixed execution path to a model where the agent evaluates specific criteria to select from a set of possible actions.

## Why is it useful?

Complex systems often need to handle a wide variety of inputs that cannot be managed by a single, linear process. A rigid workflow lacks the ability to make decisions based on context.

Routing solves this by providing a mechanism to first classify an incoming request and then direct it to the most appropriate handler. This transforms a static, predetermined execution path into a flexible and context-aware workflow capable of selecting the best possible action for a given task.

## Key Points

-   **Dynamic Decisions**: Enables agents to decide the next step in a workflow based on specific conditions.
-   **Adaptability**: Allows agents to handle diverse inputs and adapt their behavior, moving beyond simple linear execution.
-   **Implementation**: Routing logic can be implemented using LLMs (as a classifier), predefined rules, or embedding-based semantic similarity.
-   **Frameworks**: Tools like LangChain (specifically LangGraph) and the Google Agent Developer Kit (ADK) provide structured ways to define and manage routing.

## Practical Use Cases

-   **Customer Support Bots**: Classifying user queries to route them to different departments like sales, technical support, or account management.
-   **Automated Data Pipelines**: Analyzing and distributing incoming data (emails, support tickets) to the correct processing workflow based on their content or format.
-   **Multi-Agent Systems**: A central "coordinator" agent can act as a dispatcher, assigning tasks to the most suitable specialized sub-agent (e.g., a research agent, a writing agent, a coding agent).
