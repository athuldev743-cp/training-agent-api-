"""Validate the generated Module 2 JSONL and write an inspection report."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from nutrition_ingestion.models import NutritionChunk
from nutrition_ingestion.pipeline import ingest_directory

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "tests" / "fixtures" / "nutrition"
OUTPUT_DIR = ROOT / "artifacts" / "module-2"
JSONL = OUTPUT_DIR / "nutrition_chunks.jsonl"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_lines = JSONL.read_text(encoding="utf-8").splitlines()
    if not raw_lines:
        raise SystemExit("FAIL: JSONL artifact is empty")

    records: list[NutritionChunk] = []
    for line_number, line in enumerate(raw_lines, start=1):
        try:
            records.append(NutritionChunk.model_validate(json.loads(line)))
        except Exception as exc:
            raise SystemExit(f"FAIL: invalid JSONL record at line {line_number}: {exc}") from exc

    chunk_ids = [record.chunk_id for record in records]
    sources = sorted({record.source_id for record in records})
    sizes = [record.char_count for record in records]
    empty_chunks = sum(not record.text.strip() for record in records)
    invalid_char_counts = sum(record.char_count != len(record.text) for record in records)
    duplicate_ids = len(chunk_ids) - len(set(chunk_ids))
    non_nutrition_records = sum(record.domain != "nutrition" for record in records)
    missing_source_metadata = sum(not record.source_id or not record.source or not record.title for record in records)

    # Re-run the pipeline to verify deterministic output from the same fixture set.
    repeat_output = OUTPUT_DIR / ".repeat-check.jsonl"
    try:
        ingest_directory(INPUT, repeat_output)
        deterministic = repeat_output.read_bytes() == JSONL.read_bytes()
    finally:
        repeat_output.unlink(missing_ok=True)

    checks = {
        "jsonl_records_valid": len(records) == len(raw_lines),
        "no_empty_chunks": empty_chunks == 0,
        "no_duplicate_chunk_ids": duplicate_ids == 0,
        "char_counts_match_text": invalid_char_counts == 0,
        "all_records_in_nutrition_domain": non_nutrition_records == 0,
        "source_metadata_present": missing_source_metadata == 0,
        "output_is_deterministic": deterministic,
    }
    report = {
        "module": "module-2-nutrition-ingestion",
        "fixture_scope": "original Markdown and UTF-8 text fixtures; no production corpus",
        "documents_in_fixture_directory": len([
            path for path in INPUT.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}
        ]),
        "chunks_written": len(records),
        "unique_sources": sources,
        "source_counts": dict(sorted(Counter(record.source_id for record in records).items())),
        "chunk_size_chars": {
            "min": min(sizes),
            "max": max(sizes),
            "average": round(sum(sizes) / len(sizes), 2),
        },
        "empty_chunks": empty_chunks,
        "duplicate_chunk_ids": duplicate_ids,
        "invalid_char_counts": invalid_char_counts,
        "non_nutrition_records": non_nutrition_records,
        "missing_source_metadata": missing_source_metadata,
        "deterministic_output": deterministic,
        "checks": checks,
        "passed": all(checks.values()),
    }
    (OUTPUT_DIR / "inspection_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    summary = [
        "MODULE 2 NUTRITION INGESTION INSPECTION",
        f"Documents: {report['documents_in_fixture_directory']}",
        f"Chunks: {report['chunks_written']}",
        f"Unique sources: {len(sources)}",
        f"Chunk sizes (min/max/avg): {min(sizes)}/{max(sizes)}/{report['chunk_size_chars']['average']} chars",
        f"Duplicate IDs: {duplicate_ids}",
        f"Empty chunks: {empty_chunks}",
        f"Deterministic output: {'PASS' if deterministic else 'FAIL'}",
        "",
        "Checks:",
        *[f"- {'PASS' if passed else 'FAIL'}: {name}" for name, passed in checks.items()],
        "",
        f"OVERALL: {'PASS' if report['passed'] else 'FAIL'}",
        "Scope note: this is a CI fixture inspection, not a review of a real-world nutrition corpus.",
    ]
    (OUTPUT_DIR / "validation_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    if not report["passed"]:
        raise SystemExit("Module 2 artifact inspection failed")


if __name__ == "__main__":
    main()
