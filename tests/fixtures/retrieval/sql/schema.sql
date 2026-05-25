CREATE TABLE prompt_cache_entries (
  cache_key TEXT PRIMARY KEY,
  model TEXT NOT NULL,
  prompt_hash TEXT NOT NULL,
  response_body TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE embedding_vectors (
  chunk_id BIGINT PRIMARY KEY,
  source_path TEXT NOT NULL,
  vector_model TEXT NOT NULL,
  vector_blob BLOB NOT NULL
);

CREATE TABLE request_audit_log (
  request_id TEXT PRIMARY KEY,
  upstream_model TEXT NOT NULL,
  tokens_saved INTEGER NOT NULL,
  latency_ms INTEGER NOT NULL
);

CREATE INDEX idx_prompt_cache_model ON prompt_cache_entries(model);
CREATE INDEX idx_embedding_vectors_path ON embedding_vectors(source_path);
