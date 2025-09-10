import os
import asyncio
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

# --- Configuration ---
# Ensure your API key environment variable is set (e.g., OPENAI_API_KEY)

# For better security, load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

try:
   llm: Optional[ChatOpenAI] = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
  
except Exception as e:
   print(f"Error initializing language model: {e}")
   llm = None

# --- Define Independent Chains ---
# These three chains represent distinct tasks that can be executed in parallel.

summarize_chain: Runnable = (
   ChatPromptTemplate.from_messages([
       ("system", "Summarize the following topic concisely:"),
       ("user", "{topic}")
   ])
   | llm
   | StrOutputParser()
)

questions_chain: Runnable = (
   ChatPromptTemplate.from_messages([
       ("system", "Generate three interesting questions about the following topic:"),
       ("user", "{topic}")
   ])
   | llm
   | StrOutputParser()
)

terms_chain: Runnable = (
   ChatPromptTemplate.from_messages([
       ("system", "Identify 5-10 key terms from the following topic, separated by commas:"),
       ("user", "{topic}")
   ])
   | llm
   | StrOutputParser()
)

# --- Build the Parallel + Synthesis Chain ---

# 1. Define the block of tasks to run in parallel. The results of these,
#    along with the original topic, will be fed into the next step.
map_chain = RunnableParallel(
   {
       "summary": summarize_chain,
       "questions": questions_chain,
       "key_terms": terms_chain,
       "topic": RunnablePassthrough(),  # Pass the original topic through
   }
)

# 2. Define the final synthesis prompt which will combine the parallel results.
synthesis_prompt = ChatPromptTemplate.from_messages([
   ("system", """Based on the following information:
    Summary: {summary}
    Related Questions: {questions}
    Key Terms: {key_terms}
    Synthesize a comprehensive answer."""),
   ("user", "Original topic: {topic}")
])

# 3. Construct the full chain by piping the parallel results directly
#    into the synthesis prompt, followed by the LLM and output parser.
full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()

# --- Run the Chain ---
async def run_parallel_example(topic: str) -> None:
   """
   Asynchronously invokes the parallel processing chain with a specific topic
   and prints the synthesized result.

   Args:
       topic: The input topic to be processed by the LangChain chains.
   """
   if not llm:
       print("LLM not initialized. Cannot run example.")
       return

   print(f"\n--- Running Parallel LangChain Example for Topic: '{topic}' ---")
   try:
       # The input to `ainvoke` is the single 'topic' string, 
       # then passed to each runnable in the `map_chain`.
       response = await full_parallel_chain.ainvoke(topic)
       print("\n--- Final Response ---")
       print(response)
   except Exception as e:
       print(f"\nAn error occurred during chain execution: {e}")

if __name__ == "__main__":
   test_topic = "The history of space exploration"
   # In Python 3.7+, asyncio.run is the standard way to run an async function.
   asyncio.run(run_parallel_example(test_topic))
