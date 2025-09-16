"""
This script contains a collection of code examples demonstrating memory management
features within LangChain and LangGraph, as described in Chapter 8.
"""
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import (
   ChatPromptTemplate,
   MessagesPlaceholder,
   SystemMessagePromptTemplate,
   HumanMessagePromptTemplate,
)
from langgraph.store.memory import InMemoryStore

# --- Example 1: ChatMessageHistory (Manual Short-Term Memory) ---
# For direct, simple control over a conversation's history outside of a chain.
print("--- Example 1: ChatMessageHistory ---")
history = ChatMessageHistory()
history.add_user_message("I'm heading to New York next week.")
history.add_ai_message("Great! It's a fantastic city.")
print(history.messages)
print("-" * 20, "\n")


# --- Example 2: ConversationBufferMemory (Automated for Chains) ---
# Integrates memory directly into chains.
print("--- Example 2: ConversationBufferMemory with LLMChain ---")
llm = OpenAI(temperature=0)
template = """You are a helpful travel agent.
Previous conversation:
{history}
New question: {question}
Response:"""
prompt = PromptTemplate.from_template(template)
memory = ConversationBufferMemory(memory_key="history")
conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

conversation.predict(question="I want to book a flight.")
conversation.predict(question="My name is Sam, by the way.")
response = conversation.predict(question="What was my name again?")
print("Final response from LLM:", response)
print("-" * 20, "\n")


# --- Example 3: Memory with Chat Models ---
# Using return_messages=True is essential for chat models.
print("--- Example 3: ConversationBufferMemory with Chat Models ---")
chat_llm = ChatOpenAI()
chat_prompt = ChatPromptTemplate(
   messages=[
       SystemMessagePromptTemplate.from_template("You are a friendly assistant."),
       MessagesPlaceholder(variable_name="chat_history"),
       HumanMessagePromptTemplate.from_template("{question}")
   ]
)
chat_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat_conversation = LLMChain(llm=chat_llm, prompt=chat_prompt, verbose=True, memory=chat_memory)

chat_conversation.predict(question="Hi, I'm Jane.")
chat_response = chat_conversation.predict(question="Do you remember my name?")
print("Final response from Chat Model:", chat_response)
print("-" * 20, "\n")


# --- Example 4: LangGraph Long-Term Memory with InMemoryStore ---
# LangGraph stores long-term memories as JSON documents in a store.
print("--- Example 4: LangGraph InMemoryStore ---")

def embed(texts: list[str]) -> list[list[float]]:
   """Placeholder for a real embedding function."""
   return [[float(i), float(i+1)] for i, _ in enumerate(texts)]

store = InMemoryStore(index={"embed": embed, "dims": 2})

# Define a namespace (like a folder) for a user
namespace = ("user-123", "chitchat")

# 1. Put a memory into the store
print("Putting memory into store...")
store.put(
   namespace,
   "user_preferences",
   {
       "rules": [
           "User likes short, direct language",
           "User only speaks English & python",
       ],
       "my-key": "my-value",
   },
)
print("-> Done.")

# 2. Get the memory by its key
print("\nGetting memory by key...")
item = store.get(namespace, "user_preferences")
print("Retrieved Item:", item)

# 3. Search for memories using semantic search
print("\nSearching for memories...")
items = store.search(
   namespace,
   filter={"my-key": "my-value"},
   query="language preferences"
)
print("Search Results:", items)
print("-" * 20, "\n")

