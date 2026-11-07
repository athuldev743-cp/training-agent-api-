import json
from unittest.mock import MagicMock, patch

import pytest

from nutrition_ingestion.models import ExtractedPage
from nutrition_ingestion.chunk import chunk_page
from nutrition_ingestion.store import (
    EMBEDDING_DIMENSIONS,
    UPSERT_NUTRITION_SQL,
    _vector_literal,
    persist_jsonl,
)


def _write_jsonl(path, count=2):
    page = ExtractedPage(
        source_id="guide.md", source="guide.md", title="Guide",
        text="Protein supports recovery and muscle repair. " * count,
    )
    chunks = chunk_page(page, max_chars=300, overlap=30)
    path.write_text(
        "\n".join(chunk.model_dump_json() for chunk in chunks) + "\n",
        encoding="utf-8",
    )
    return chunks


def test_vector_literal_rejects_wrong_dimension_and_non_finite_values():
    with pytest.raises(ValueError, match="384 dimensions"):
        _vector_literal([0.1, 0.2])
    with pytest.raises(ValueError, match="finite"):
        _vector_literal([float("nan")] * EMBEDDING_DIMENSIONS)


def test_upsert_is_nutrition_only_and_idempotent():
    assert "INSERT INTO nutrition_knowledge" in UPSERT_NUTRITION_SQL
    assert "ON CONFLICT (chunk_id)" in UPSERT_NUTRITION_SQL
    assert "fitness_knowledge" not in UPSERT_NUTRITION_SQL


def test_persist_jsonl_embeds_and_writes_batches(tmp_path):
    source = tmp_path / "chunks.jsonl"
    chunks = _write_jsonl(source)
    model = MagicMock()
    model.embed.return_value = iter([[0.01] * EMBEDDING_DIMENSIONS for _ in chunks])
    connection = MagicMock()
    cursor = connection.__enter__.return_value.cursor.return_value.__enter__.return_value
    cursor.rowcount = len(chunks)

    with patch("nutrition_ingestion.store.get_connection", return_value=connection):
        report = persist_jsonl(source, batch_size=1, embedding_model=model)

    assert report["records_read"] == len(chunks)
    assert report["records_persisted_or_reembedded"] == len(chunks)
    assert model.embed.call_count == len(chunks)
    assert cursor.executemany.call_count == len(chunks)


def test_persist_rejects_empty_file_before_database_connection(tmp_path):
    source = tmp_path / "empty.jsonl"
    source.write_text("", encoding="utf-8")
    with patch("nutrition_ingestion.store.get_connection") as connect:
        with pytest.raises(ValueError, match="empty"):
            persist_jsonl(source, embedding_model=MagicMock())
    connect.assert_not_called()


def test_persist_rejects_wrong_embedding_count_and_rolls_back(tmp_path):
    source = tmp_path / "chunks.jsonl"
    _write_jsonl(source)
    model = MagicMock()
    model.embed.return_value = iter([])
    connection = MagicMock()
    with patch("nutrition_ingestion.store.get_connection", return_value=connection):
        with pytest.raises(RuntimeError, match="unexpected number"):
            persist_jsonl(source, embedding_model=model)
    connection.__exit__.assert_not_called()


def test_persist_rejects_invalid_jsonl(tmp_path):
    source = tmp_path / "invalid.jsonl"
    source.write_text(json.dumps({"domain": "fitness"}) + "\n", encoding="utf-8")
    with patch("nutrition_ingestion.store.get_connection") as connect:
        with pytest.raises(ValueError, match="invalid nutrition JSONL"):
            persist_jsonl(source, embedding_model=MagicMock())
    connect.assert_not_called()
