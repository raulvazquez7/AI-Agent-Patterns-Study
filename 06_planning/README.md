# 06. Planning

## What is the Planning Pattern?

The Planning pattern enables an agent to break down complex goals into a sequence of smaller, manageable steps. Instead of reacting to immediate input, the agent formulates a plan to move from an initial state towards a desired outcome. This is crucial for tasks that cannot be solved with a single action and require foresight and strategy.

The agent's core task is to autonomously chart a course to the goal. It must understand the initial state, the goal state, and then discover the optimal sequence of actions to connect them.

## Why is it useful?

Planning transforms a simple reactive agent into a strategic executor. It is a core process in autonomous systems, allowing them to:

-   **Handle Complexity**: Decompose high-level objectives into a structured plan composed of discrete, executable steps.
-   **Automate Workflows**: Orchestrate multi-step processes like onboarding an employee or generating a research report, managing dependencies in a logical order.
-   **Adapt to Obstacles**: An initial plan is a starting point. A capable agent can incorporate new information (e.g., a preferred venue is unavailable) and re-evaluate its options to formulate a new plan.
-   **Synthesize Information**: For tasks like creating a report, the agent can plan distinct phases for information gathering, summarization, structuring, and refinement.

Use this pattern when a request is too complex for a single tool and requires a sequence of interdependent operations to reach a final, synthesized outcome.

## A Rule of Thumb: When to Use Planning

To decide whether to use this pattern, ask yourself one key question: **Does the 'how' need to be discovered, or is it already known?**

-   **Use Dynamic Planning** when the path to the goal is not obvious and requires adaptation. This is ideal for complex, unpredictable tasks like planning a trip or conducting research, where the agent must create and adjust its plan based on new information.

-   **Use a Fixed Workflow** for repetitive, predictable tasks where the steps are always the same (e.g., processing a standardized form). In these cases, a dynamic planner is inefficient. A simple chain or a graph with a predefined path is a better and more reliable choice.
