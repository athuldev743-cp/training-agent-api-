import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from nutrition_ingestion.chunk import chunk_page
from nutrition_ingestion.extract import DocumentExtractionError, extract_document
from nutrition_ingestion.models import ExtractedPage, NutritionChunk
from nutrition_ingestion.pipeline import ingest_directory

def test_chunking_respects_size_overlap_and_page_metadata():
    page = ExtractedPage(source_id="guidelines.txt", source="guidelines.txt", title="Guidelines",
                         page=3, text="Protein supports muscle repair and growth. " * 30)
    chunks = chunk_page(page, max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 and c.page == 3 and c.domain == "nutrition" for c in chunks)
    assert all(c.char_count == len(c.text) for c in chunks)
    assert chunks[0].chunk_number == 0 and chunks[1].chunk_number == 1

def test_chunk_ids_are_deterministic():
    page = ExtractedPage(source_id="guide.md", source="guide.md", title="Guide",
                         text="A balanced diet includes a variety of foods.")
    assert [x.chunk_id for x in chunk_page(page)] == [x.chunk_id for x in chunk_page(page)]

def test_chunker_rejects_invalid_overlap():
    page = ExtractedPage(source_id="guide.md", source="guide.md", title="Guide", text="Nutrition basics.")
    with pytest.raises(ValueError):
        chunk_page(page, max_chars=100, overlap=100)

def test_chunk_contract_rejects_invalid_hash_and_domain():
    with pytest.raises(ValidationError):
        NutritionChunk(chunk_id="bad", domain="fitness", source_id="a", source="a",
                       title="A", chunk_number=0, text="text", char_count=4, token_estimate=1)

def test_text_extraction_preserves_source_and_text(tmp_path: Path):
    root = tmp_path / "sources"
    root.mkdir()
    file = root / "guide.txt"
    file.write_text("Protein supports recovery.\n\nHydration matters.", encoding="utf-8")
    pages, empty = extract_document(file, root)
    assert empty == 0 and len(pages) == 1
    assert pages[0].source_id == "guide.txt" and pages[0].page is None
    assert "Hydration matters." in pages[0].text

def test_extraction_rejects_path_outside_source_root(tmp_path: Path):
    root = tmp_path / "sources"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("not in source root", encoding="utf-8")
    with pytest.raises(DocumentExtractionError):
        extract_document(outside, root)

def test_pipeline_writes_valid_repeatable_jsonl(tmp_path: Path):
    root = tmp_path / "sources"
    root.mkdir()
    (root / "guide.md").write_text("# Nutrition\n\n" + "Whole foods provide nutrients. " * 40, encoding="utf-8")
    output = tmp_path / "build" / "nutrition.jsonl"
    report = ingest_directory(root, output, max_chars=300, overlap=40)
    content = output.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in content.splitlines()]
    assert report.files_processed == 1 and report.chunks_written == len(rows) > 1
    assert all(row["domain"] == "nutrition" and row["source_id"] == "guide.md" for row in rows)
    second = ingest_directory(root, output, max_chars=300, overlap=40)
    assert second.chunks_written == report.chunks_written
    assert output.read_text(encoding="utf-8") == content

def test_pipeline_rejects_empty_input_without_output(tmp_path: Path):
    root = tmp_path / "sources"
    root.mkdir()
    output = tmp_path / "build" / "nutrition.jsonl"
    with pytest.raises(ValueError, match="no supported"):
        ingest_directory(root, output)
    assert not output.exists()

def test_pipeline_rejects_output_inside_input(tmp_path: Path):
    root = tmp_path / "sources"
    root.mkdir()
    (root / "guide.txt").write_text("Nutrition document.", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        ingest_directory(root, root / "chunks.jsonl")


def test_extraction_rejects_empty_text_document(tmp_path: Path):
    root = tmp_path / "sources"
    root.mkdir()
    empty_file = root / "empty.txt"
    empty_file.write_text("   \\n\\t", encoding="utf-8")
    with pytest.raises(DocumentExtractionError, match="no extractable text"):
        extract_document(empty_file, root)


def test_extraction_rejects_corrupt_pdf(tmp_path: Path):
    root = tmp_path / "sources"
    root.mkdir()
    corrupt_file = root / "corrupt.pdf"
    corrupt_file.write_bytes(b"this is not a valid PDF")
    with pytest.raises(DocumentExtractionError):
        extract_document(corrupt_file, root)
