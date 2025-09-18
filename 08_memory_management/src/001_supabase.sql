
-- 1. Habilitar la extensión pgvector
create extension if not exists vector;

-- 2. Crear la tabla para el historial, ahora con 768 dimensiones para el modelo de Google
create table
  conversation_history (
    id bigserial primary key,
    created_at timestamp with time zone default now() not null,
    thread_id text not null,
    content text not null,
    embedding vector (768)
  );

-- 3. Crear el índice HNSW para búsquedas de alta velocidad
-- Usamos 'vector_cosine_ops' que es ideal para la búsqueda de similitud semántica.
CREATE INDEX ON conversation_history
USING hnsw (embedding vector_cosine_ops);

-- 4. Crear la función de búsqueda, adaptada a 768 dimensiones
create function match_conversation_history (
  query_embedding vector(768),
  match_thread_id text,
  match_count int
) returns table (
  id bigint,
  content text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    conversation_history.id,
    conversation_history.content,
    1 - (conversation_history.embedding <=> query_embedding) as similarity
  from
    conversation_history
  where
    conversation_history.thread_id = match_thread_id
  and 1 - (conversation_history.embedding <=> query_embedding) > 0.6
  order by
    similarity desc
  limit
    match_count;
end;
$$;