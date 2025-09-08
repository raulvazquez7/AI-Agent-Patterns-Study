# Prompt Chaining

## What is Prompt Chaining?

**Prompt Chaining**, also known as the *Pipeline* pattern, is a technique for tackling complex tasks using large language models (LLMs). Instead of trying to solve a problem with a single, complex prompt, this strategy applies the "divide and conquer" principle.

The core idea is to break down the problem into a sequence of smaller, more manageable sub-problems. Each sub-problem is solved with a specific prompt, and the output of one prompt is used as the input for the next, creating a logical chain.

## Why is it useful?

Using a single prompt for multifaceted tasks can be inefficient and prone to errors such as:

- **Instruction neglect**: The model ignores parts of the prompt.
- **Contextual drift**: The model deviates from the initial goal.
- **Error propagation**: An early error affects the entire result.
- **Hallucinations**: The cognitive load increases the probability of generating incorrect information.

*Prompt Chaining* solves these problems by creating a sequential and focused workflow, which significantly improves reliability and control over the final output.

## Key Points

- **Decomposition**: Breaks down complex tasks into smaller, focused steps.
- **Sequential Flow**: The output of one step is the input for the next, maintaining context.
- **Improved Reliability**: Reduces the model's cognitive load, leading to more accurate and controlled results.
- **Modularity**: Facilitates debugging and optimization of each step individually.
- **Integration**: Allows the incorporation of external tools (APIs, databases) between the steps of the chain.

## Practical Use Case

A common example is information processing:

1.  **Prompt 1 (Extraction)**: Extract technical specifications from unstructured text.
2.  **Prompt 2 (Transformation)**: Convert those specifications into a structured format like JSON.

This approach ensures that each task is performed accurately before moving on to the next, guaranteeing a robust and correct final result. It is a fundamental technique for building more sophisticated AI systems and agents capable of multi-step reasoning.
