# 08. Memory Management

## What is the Memory Management Pattern?

Memory Management enables an agent to retain and utilize information from past interactions, observations, and learning experiences. This is essential for agents to maintain conversational context, make informed decisions, and improve over time.

Memory is typically categorized into two main types:

-   **Short-Term Memory (Contextual Memory)**: This is the agent's working memory, holding information from the current interaction. In LLM-based agents, this is often managed within the model's "context window." It's ephemeral and is lost once a session ends.
-   **Long-Term Memory (Persistent Memory)**: This is a repository for information that needs to be retained across different sessions. It's stored externally in databases, knowledge graphs, or vector stores, allowing for semantic search and retrieval of past knowledge.

## Why is it useful?

Effective memory management elevates an agent from a simple Q&A bot to an intelligent assistant. It is critical for:

-   **Maintaining Context**: Ensuring conversations are coherent and natural.
-   **Personalization**: Recalling user preferences and past interactions to provide a tailored experience.
-   **Multi-Step Tasks**: Tracking progress and intermediate results in complex workflows.
-   **Learning and Adaptation**: Storing successful strategies or mistakes to improve future performance.

## Implementations: Google ADK vs. LangGraph

-   **Google ADK** provides a structured approach with three core components:
    -   **Session**: Represents a single conversation thread, logging all messages and actions.
    -   **State (`session.state`)**: A dictionary within a `Session` for temporary, session-specific data (the "scratchpad").
    -   **MemoryService**: Manages the connection to a searchable, long-term knowledge base.

-   **LangChain & LangGraph** offer flexible tools:
    -   **`ConversationBufferMemory`**: A common way to automatically load the history of a single conversation into a prompt.
    -   **Checkpointers**: LangGraph uses checkpointers to automatically save and resume the state of a conversation, effectively managing short-term memory.
    -   **Stores (`BaseStore`)**: LangGraph provides interfaces for connecting to external databases to manage long-term, persistent memory across different users and sessions.

## Rule of Thumb

Use this pattern when an agent needs to do more than answer a single, isolated question. It is essential for any agent that must maintain context in a conversation, track progress in a task, or personalize interactions by recalling user history.
