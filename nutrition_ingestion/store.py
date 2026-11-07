"""Generate embeddings and persist validated nutrition chunks to PostgreSQL."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Any

from fastembed import TextEmbedding

from nutrition_ingestion.models import NutritionChunk
from rag.neon import get_connection

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
MAX_BATCH_SIZE = 128

UPSERT_NUTRITION_SQL = """
INSERT INTO nutrition_knowledge (
    chunk_id, domain, source_id, source, title, page, chunk_number,
    content, embedding, embedding_model, updated_at
)
VALUES (%s, 'nutrition', %s, %s, %s, %s, %s, %s, %s::vector, %s, now())
ON CONFLICT (chunk_id) DO UPDATE SET
    embedding = EXCLUDED.embedding,
    embedding_model = EXCLUDED.embedding_model,
    updated_at = now()
WHERE nutrition_knowledge.embedding IS NULL
   OR nutrition_knowledge.embedding_model IS DISTINCT FROM EXCLUDED.embedding_model
"""


def _vector_literal(values: Iterable[Any]) -> str:
    vector = [float(value) for value in values]
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(f"embedding must contain {EMBEDDING_DIMENSIONS} dimensions")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedding values must be finite")
    return "[" + ",".join(format(value, ".8g") for value in vector) + "]"


def _read_batch(handle, batch_size: int) -> list[NutritionChunk]:
    batch = []
    while len(batch) < batch_size:
        line = handle.readline()
        if not line:
            break
        if not line.strip():
            continue
        try:
            batch.append(NutritionChunk.model_validate_json(line))
        except Exception as exc:
            raise ValueError("invalid nutrition JSONL record") from exc
    return batch


def persist_jsonl(
    input_file: Path,
    *,
    batch_size: int = 32,
    embedding_model: TextEmbedding | None = None,
) -> dict[str, int | str]:
    """Persist JSONL chunks with embeddings; one DB transaction covers the full file."""
    path = Path(input_file)
    if path.is_symlink() or not path.is_file():
        raise ValueError("input_file must be an existing regular file")
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}")

    model = embedding_model or TextEmbedding(model_name=EMBEDDING_MODEL)
    total = 0
    inserted_or_updated = 0

    with path.open("r", encoding="utf-8") as handle:
        # Validate that the file contains at least one record before opening a DB connection.
        first_record = next((line for line in handle if line.strip()), None)
        if first_record is None:
            raise ValueError("nutrition JSONL file is empty")
        try:
            NutritionChunk.model_validate_json(first_record)
        except Exception as exc:
            raise ValueError("invalid nutrition JSONL record") from exc
        handle.seek(0)

        with get_connection() as connection:
            with connection.cursor() as cursor:
                while True:
                    batch = _read_batch(handle, batch_size)
                    if not batch:
                        break
                    vectors = list(model.embed([chunk.text for chunk in batch]))
                    if len(vectors) != len(batch):
                        raise RuntimeError("embedding model returned an unexpected number of vectors")
                    rows = []
                    for chunk, vector in zip(batch, vectors):
                        rows.append((
                            chunk.chunk_id, chunk.source_id, chunk.source, chunk.title,
                            chunk.page, chunk.chunk_number, chunk.text,
                            _vector_literal(vector), EMBEDDING_MODEL,
                        ))
                    cursor.executemany(UPSERT_NUTRITION_SQL, rows)
                    affected = max(cursor.rowcount, 0)
                    inserted_or_updated += affected
                    total += len(batch)

    return {
        "records_read": total,
        "records_persisted_or_reembedded": inserted_or_updated,
        "records_unchanged": total - inserted_or_updated,
        "embedding_model": EMBEDDING_MODEL,
    }
