-- Module 3: isolated nutrition retrieval corpus.
-- Run once against Neon PostgreSQL with the pgvector extension available.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS nutrition_knowledge (
    chunk_id CHAR(64) PRIMARY KEY,
    domain TEXT NOT NULL DEFAULT 'nutrition' CHECK (domain = 'nutrition'),
    source_id TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    page INTEGER CHECK (page IS NULL OR page >= 1),
    chunk_number INTEGER NOT NULL CHECK (chunk_number >= 0),
    content TEXT NOT NULL CHECK (length(btrim(content)) > 0),
    embedding VECTOR(384),
    content_tsv TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS nutrition_knowledge_embedding_hnsw_idx
    ON nutrition_knowledge USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS nutrition_knowledge_content_tsv_idx
    ON nutrition_knowledge USING gin (content_tsv);

CREATE INDEX IF NOT EXISTS nutrition_knowledge_source_idx
    ON nutrition_knowledge (source_id);
