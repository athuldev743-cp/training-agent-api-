-- Track the embedding model used for each nutrition chunk.
-- Safe to apply after migration 003 has created nutrition_knowledge.
ALTER TABLE nutrition_knowledge
    ADD COLUMN IF NOT EXISTS embedding_model TEXT;

UPDATE nutrition_knowledge
SET embedding_model = 'BAAI/bge-small-en-v1.5'
WHERE embedding IS NOT NULL AND embedding_model IS NULL;
