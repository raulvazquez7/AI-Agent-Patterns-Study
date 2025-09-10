# 04. Reflection

## What is Reflection?

The **Reflection** pattern involves an agent evaluating its own work to improve its performance or refine its response. It's a form of self-correction that introduces a feedback loop into the workflow. Instead of just producing a final output, the agent examines that output, identifies potential issues, and uses those insights to generate a better version.

A highly effective implementation is the **Producer-Critic** model, where one agent (the Producer) generates the content, and a separate agent (the Critic) evaluates it against specific criteria. This separation of concerns prevents the "cognitive bias" of an agent reviewing its own work and often leads to more robust results.

## Why is it useful?

An agent's initial output might not be optimal, accurate, or complete. Basic workflows lack a mechanism for the agent to recognize and fix its own errors. The Reflection pattern provides a structured way to iteratively improve the quality of the final output, ensuring it meets higher standards of accuracy, coherence, and adherence to complex instructions.

## Key Points

-   **Iterative Improvement**: The core is a feedback loop of execution, evaluation/critique, and refinement.
-   **Higher Quality Output**: Leads to more accurate, polished, and reliable results.
-   **Producer-Critic Model**: A powerful implementation that separates the generation and evaluation roles for better objectivity.
-   **Trade-offs**: This pattern increases latency and cost due to the necessity of multiple LLM calls for each refinement cycle.
-   **State Management**: True iterative reflection often requires more complex orchestration with state management, as seen in frameworks like LangGraph.

## Practical Use Cases

-   **Content Generation**: Refining a draft of a blog post for flow, tone, and clarity.
-   **Code Generation**: Writing initial code, then using a critic agent to identify bugs, style issues, or missing edge cases, and finally refining the code.
-   **Complex Problem-Solving**: Evaluating a proposed plan, identifying flaws, and revising the strategy before execution.
-   **Summarization**: Generating an initial summary and then comparing it against the source document to ensure accuracy and completeness.
