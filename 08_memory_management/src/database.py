import logging
from supabase import Client as SupabaseClient
from langchain_core.embeddings import Embeddings
# CORREGIDO: Importar 'traceable' desde la librería langsmith
from langsmith import traceable

@traceable
def add_turn_to_vector_memory(
    supabase_client: SupabaseClient,
    thread_id: str,
    turn_content: str,
    embedding_model: Embeddings
):
    """
    Vectorizes a conversation turn and stores it in the Supabase vector store.
    """
    logging.info(f"--- Updating vector memory for thread {thread_id} ---")
    try:
        embedding = embedding_model.embed_query(turn_content)
        
        supabase_client.table("conversation_history").insert({
            "thread_id": thread_id,
            "content": turn_content,
            "embedding": embedding
        }).execute()
        
        logging.info("--- Vector memory updated successfully. ---")
    except Exception as e:
        logging.error(f"Error updating vector memory: {e}", exc_info=True)

@traceable
def search_vector_memory(
    supabase_client: SupabaseClient,
    thread_id: str,
    query: str,
    embedding_model: Embeddings,
    k: int = 3
) -> list[str]:
    """
    Searches the vector memory for the most relevant past conversation turns.
    """
    logging.info(f"--- Searching vector memory for: '{query}' ---")
    try:
        query_embedding = embedding_model.embed_query(query)
        
        results = supabase_client.rpc("match_conversation_history", {
            "query_embedding": query_embedding,
            "match_thread_id": thread_id,
            "match_count": k
        }).execute()
        
        if not results.data:
            return []
            
        return [item['content'] for item in results.data]
        
    except Exception as e:
        logging.error(f"Error searching vector memory: {e}", exc_info=True)
        return []
