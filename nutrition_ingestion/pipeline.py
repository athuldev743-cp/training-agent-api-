"""Run deterministic nutrition extraction and JSONL export."""
import os
import tempfile
from pathlib import Path
from nutrition_ingestion.chunk import chunk_page
from nutrition_ingestion.extract import SUPPORTED_SUFFIXES, extract_document
from nutrition_ingestion.models import IngestionReport, NutritionChunk

def ingest_directory(input_dir: Path, output_file: Path, *, max_chars: int = 1200, overlap: int = 200) -> IngestionReport:
    """Write validated chunks atomically. Embeddings and DB writes belong to later modules."""
    root = Path(input_dir).resolve()
    output = Path(output_file).resolve()
    if not root.is_dir():
        raise ValueError(f"input directory does not exist: {root}")
    if output == root or root in output.parents:
        raise ValueError("output file must be outside the input directory")
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise ValueError(f"no supported PDF, TXT or Markdown documents found in {root}")

    chunks: list[NutritionChunk] = []
    pages_with_text = 0
    empty_pages = 0
    for path in files:
        extracted, empty_count = extract_document(path, root)
        pages_with_text += len(extracted)
        empty_pages += empty_count
        for page in extracted:
            chunks.extend(chunk_page(page, max_chars=max_chars, overlap=overlap))
    if not chunks:
        raise ValueError("ingestion produced zero chunks; no output was written")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for chunk in chunks:
                handle.write(chunk.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return IngestionReport(
        files_processed=len(files), pages_with_text=pages_with_text,
        empty_pages=empty_pages, chunks_written=len(chunks), output_path=str(output),
    )
