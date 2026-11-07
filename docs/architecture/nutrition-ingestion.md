# Module 2 — Nutrition document ingestion

## Purpose
Convert curated nutrition sources into validated, reproducible JSONL chunk records.
This module does not generate embeddings, write to PostgreSQL, or serve API
requests. Those concerns are deferred to the retrieval module.

## Pipeline
1. Discover PDF, TXT and Markdown files under a configured source directory.
2. Extract UTF-8 text or page-level PDF text while retaining source and page.
3. Normalize whitespace and line endings.
4. Split each page independently into bounded chunks with configurable overlap.
5. Validate every chunk with Pydantic.
6. Generate deterministic SHA-256 IDs from source, page, chunk index and content.
7. Atomically write JSONL so a failed run cannot leave a partial output.

PDF extraction is text-based. Scanned/image-only PDFs are rejected; OCR is not
silently assumed. Add OCR only if the source collection requires it and its
quality can be evaluated.

## Data isolation
Every record has domain="nutrition" and a source-relative source_id.
Nutrition output is a separate artifact and must never be loaded into the
fitness_knowledge corpus. Module 3 will add a dedicated nutrition retrieval
table/schema and embeddings.

## Dependency boundary
pypdf is an optional ingestion extra, not part of the hosted API runtime.
CI installs the extra to test ingestion; the API host continues using base
dependencies only.

## Run locally
Install the optional dependency:
    python -m pip install -e ".[ingestion]"
Place approved documents under nutrition/source_docs/ and run:
    python -m nutrition_ingestion.cli --input nutrition/source_docs --output build/nutrition_chunks.jsonl

Defaults are 1,200 characters per chunk and 200 characters of overlap. Review
representative chunk boundaries before changing these values.

## Evaluation
Tests cover metadata, chunk bounds, deterministic IDs, validation, path
containment, JSONL integrity and repeatability. Corpus review must also inspect
tables, headings, duplicated boilerplate and context split across boundaries.
