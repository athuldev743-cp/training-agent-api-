"""Build reviewable nutrition RAG candidates from page-level PDF extraction.

The builder is deliberately separate from ingestion: it never writes to Neon or
changes the existing nutrition corpus. Complex tables/charts remain review-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "nutrition" / "inspection" / "combined_extraction" / "pdf_pages_raw.jsonl"
DEFAULT_OUTPUT_ROOT = ROOT / "nutrition" / "processed"

TARGET_CHARS = 1100
MAX_CHARS = 1450
MIN_CHARS = 180

SOURCE_CONFIG = {
    "icmr_dietary_guidelines_2024": {
        "title": "ICMR-NIN Dietary Guidelines 2024",
        "exclude_pages": {148},
        "knowledge_type": "guideline",
    },
    "icmr_rda_ear_2020": {
        "title": "ICMR-NIN RDA/EAR 2020",
        "exclude_pages": set(),
        "knowledge_type": "nutrient_requirement",
    },
    "issn_protein_exercise_2017": {
        "title": "ISSN Position Stand: Protein and Exercise (2017)",
        "include_pages": set(range(1, 20)),
        "knowledge_type": "scientific_evidence",
    },
}

KNOWN_COMPLEX_PAGES = {
    "icmr_dietary_guidelines_2024": {
        15, 16, 22, 23, 24, 25, 26, 27, 28, 33, 34, 35, 36,
        46, 47, 51, 59, 60, 61, 62, 70, 75, 77, 81, 84, 87,
        88, 101, 103, 113, 114, 115, 119, 125, 130, 131, 132,
        133, 134, 135, 136, 137, 138, 139, 142, 143, 146,
    },
    "icmr_rda_ear_2020": {4, 5, 6},
    "issn_protein_exercise_2017": set(),
}

ICMR_EXAMPLE_CHART_PAGES = set(range(130, 147))

SECTION_RE = re.compile(
    r"^(?:(?:chapter|section)\s+[\w.-]+|\d+(?:\.\d+)*\.?\s+\S.{2,100})$",
    re.IGNORECASE,
)


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if SECTION_RE.match(stripped):
        return True
    letters = [char for char in stripped if char.isalpha()]
    return bool(
        4 <= len(stripped) <= 90
        and letters
        and (
            all(char == char.upper() for char in letters)
            or (stripped == stripped.title() and " " in stripped)
        )
        and re.fullmatch(r"[A-Za-z0-9 ,:&()/\-]+", stripped)
    )


MOJIBAKE_MARKERS = ("â€“", "â€”", "â€™", "â€œ", "â€", "Ã", "Â", "\ufffd")
BULLET_TRANSLATION = str.maketrans({
    "\x01": "• ", "\x02": "• ", "\uf0b7": "• ", "\u2022": "• ",
})


def repair_mojibake(value: str) -> str:
    """Repair common UTF-8-as-Latin-1 extraction corruption only when improved."""
    if not any(marker in value for marker in MOJIBAKE_MARKERS):
        return value
    try:
        candidate = value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    before = sum(value.count(marker) for marker in MOJIBAKE_MARKERS)
    after = sum(candidate.count(marker) for marker in MOJIBAKE_MARKERS)
    return candidate if after < before else value


def normalize_text(value: str) -> str:
    value = repair_mojibake(value).translate(BULLET_TRANSLATION)
    value = value.replace("\u00ad", "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", value)
    # Preserve multi-space separators so table columns remain detectable.
    value = re.sub(r"[ \t]{2,}", "    ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def block_order(blocks: list[dict[str, Any]], page_width: float) -> list[dict[str, Any]]:
    """Order prose blocks down the left column, then the right, retaining full-width blocks."""
    valid = [b for b in blocks if normalize_text(str(b.get("text", "")))]
    if not valid:
        return []
    centers = [((b.get("bbox") or [0, 0, 0, 0])[0] +
                (b.get("bbox") or [0, 0, 0, 0])[2]) / 2 for b in valid]
    left = sum(x < page_width * 0.48 for x in centers)
    right = sum(x > page_width * 0.52 for x in centers)
    if left < 3 or right < 3:
        return sorted(valid, key=lambda b: (
            (b.get("bbox") or [0, 0, 0, 0])[1],
            (b.get("bbox") or [0, 0, 0, 0])[0],
            b.get("block_index", 0),
        ))

    spanning, left_col, right_col = [], [], []
    for block in valid:
        x0, y0, x1, _ = block.get("bbox") or [0, 0, 0, 0]
        center = (x0 + x1) / 2
        if x1 - x0 >= page_width * 0.58:
            spanning.append(block)
        elif center < page_width * 0.5:
            left_col.append(block)
        else:
            right_col.append(block)
    columns = (
        sorted(left_col, key=lambda b: (b["bbox"][1], b["bbox"][0])) +
        sorted(right_col, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    )
    result = []
    for span in sorted(spanning, key=lambda b: b["bbox"][1]):
        while columns and columns[0]["bbox"][1] <= span["bbox"][1]:
            result.append(columns.pop(0))
        result.append(span)
    result.extend(columns)
    return result


def _is_table_row(line: str) -> bool:
    return bool(
        line.count("|") >= 2
        or "\t" in line
        or re.search(r"\S\s{2,}\S\s{2,}\S", line)
    )


def split_into_units(text: str) -> list[str]:
    """Keep headings with following prose and contiguous table rows together."""
    text = normalize_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    raw_units: list[str] = []
    for paragraph in paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) <= 1:
            raw_units.append(paragraph)
            continue

        prose: list[str] = []
        table_rows: list[str] = []
        for line in lines:
            if _is_table_row(line):
                if prose:
                    raw_units.append(" ".join(prose))
                    prose = []
                table_rows.append(line)
                continue
            if table_rows:
                raw_units.append("\n".join(table_rows))
                table_rows = []
            if _is_heading(line):
                if prose:
                    raw_units.append(" ".join(prose))
                    prose = []
                raw_units.append(line)
            else:
                prose.append(line)
        if table_rows:
            raw_units.append("\n".join(table_rows))
        if prose:
            raw_units.append(" ".join(prose))

    # A heading is context for the next evidence unit, not an orphan chunk.
    units: list[str] = []
    pending_heading = ""
    for unit in raw_units:
        if _is_heading(unit) and len(unit) <= 120:
            pending_heading = f"{pending_heading}\n{unit}".strip()
            continue
        if pending_heading:
            unit = f"{pending_heading}\n{unit}"
            pending_heading = ""
        units.append(unit)
    if pending_heading:
        if units:
            units[-1] = f"{units[-1]}\n{pending_heading}"
        else:
            units.append(pending_heading)
    return units


def _split_oversized_unit(unit: str, limit: int = MAX_CHARS) -> list[str]:
    """Split long prose on sentence/word boundaries; never drop text."""
    if len(unit) <= limit:
        return [unit]
    lines = unit.splitlines()
    if len(lines) > 1 and all(_is_table_row(line) for line in lines[1:]):
        # Repeat both the section title and column header when splitting a long table.
        prefix = [lines[0]]
        row_start = 1
        if not _is_table_row(lines[0]) and len(lines) > 2:
            prefix.append(lines[1])
            row_start = 2
        groups: list[str] = []
        current = list(prefix)
        for row in lines[row_start:]:
            current_size = sum(map(len, current)) + max(0, len(current) - 1)
            if current_size + len(row) + 1 > limit and len(current) > len(prefix):
                groups.append("\n".join(current))
                current = list(prefix)
            current.append(row)
        if current:
            groups.append("\n".join(current))
        return groups

    sentences = re.split(r"(?<=[.!?])\s+", unit)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > limit:
            words = sentence.split()
            word_piece = ""
            for word in words:
                candidate = f"{word_piece} {word}".strip()
                if len(candidate) > limit and word_piece:
                    if current:
                        pieces.append(current)
                        current = ""
                    pieces.append(word_piece)
                    word_piece = word
                else:
                    word_piece = candidate
            if word_piece:
                sentence = word_piece
        candidate = f"{current} {sentence}".strip()
        if len(candidate) > limit and current:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def make_chunks(units: list[str]) -> list[str]:
    """Pack semantic units to target size, splitting only oversized units safely."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for original in units:
        for unit in _split_oversized_unit(original):
            if len(unit) > MAX_CHARS:
                raise ValueError("oversized unit could not be safely split")
            proposed = current_len + (2 if current else 0) + len(unit)
            if current and proposed > TARGET_CHARS:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(unit)
            current_len += (2 if len(current) > 1 else 0) + len(unit)
    if current:
        chunks.append("\n\n".join(current))
    if len(chunks) > 1 and len(chunks[-1]) < MIN_CHARS:
        merged = chunks[-2] + "\n\n" + chunks[-1]
        if len(merged) <= MAX_CHARS:
            chunks[-2:] = [merged]
    return chunks


def classify_chunk(text: str, source_id: str, page: int) -> str:
    lower = text.lower()
    if source_id == "icmr_rda_ear_2020" or any(
        term in lower for term in (
            "recommended dietary allowance", "estimated average requirement",
            "nutrient requirement", "rda", "ear",
        )
    ):
        return "nutrient_requirement"
    if source_id == "issn_protein_exercise_2017":
        return "scientific_evidence"
    if source_id == "icmr_dietary_guidelines_2024" and page in ICMR_EXAMPLE_CHART_PAGES:
        return "illustrative_food_chart_reference_only"
    if any(term in lower for term in ("position stand", "we recommend", "recommendation")):
        return "recommendation"
    return SOURCE_CONFIG[source_id]["knowledge_type"]


def stable_id(source_id: str, page: int, chunk_number: int, content: str) -> str:
    payload = f"{source_id}|{page}|{chunk_number}|{content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_candidates(input_path: Path, output_root: Path, run_id: str | None = None) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Extraction file not found: {input_path}")
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"new_sources_run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates_path = output_dir / "nutrition_chunk_candidates.jsonl"
    review_path = output_dir / "nutrition_chunk_review_queue.jsonl"
    page_count, chunk_count, review_count, source_chars = Counter(), Counter(), Counter(), Counter()
    seen_ids: set[str] = set()

    with input_path.open("r", encoding="utf-8") as source_file, \
         candidates_path.open("w", encoding="utf-8") as candidate_file, \
         review_path.open("w", encoding="utf-8") as review_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on extraction line {line_number}") from exc
            source_id = row.get("source_id")
            if source_id not in SOURCE_CONFIG:
                continue
            page = int(row["page"])
            config = SOURCE_CONFIG[source_id]
            if page in config.get("exclude_pages", set()):
                continue
            if "include_pages" in config and page not in config["include_pages"]:
                continue

            page_count[source_id] += 1
            source_chars[source_id] += int(row.get("text_chars_sorted", 0))
            blocks = row.get("blocks") or []
            ordered = block_order(blocks, float(row.get("page_width", 0)))
            block_text = "\n\n".join(
                normalize_text(str(block.get("text", ""))) for block in ordered
                if normalize_text(str(block.get("text", "")))
            )
            page_text = block_text or normalize_text(str(row.get("text_sorted", "")))
            if not page_text:
                continue

            flags = []
            if page in KNOWN_COMPLEX_PAGES.get(source_id, set()):
                flags.append("known_table_or_chart_page")
            if int(row.get("image_count", 0)) >= 8:
                flags.append("image_rich_page")
            if int(row.get("vector_drawing_count", 0)) >= 250:
                flags.append("vector_heavy_page")
            if len(page_text) < 120:
                flags.append("low_text_page")
            if any(marker in page_text for marker in MOJIBAKE_MARKERS):
                flags.append("mojibake_unresolved")
            if source_id == "icmr_dietary_guidelines_2024" and page in ICMR_EXAMPLE_CHART_PAGES:
                flags.append("illustrative_chart_reference_only")
                flags.append("do_not_present_as_fixed_meal_plan")
            if source_id == "icmr_rda_ear_2020" and page in {4, 5, 6}:
                flags.append("rda_ear_table_atomic_review")

            page_chunks = make_chunks(split_into_units(page_text))
            for chunk_number, content in enumerate(page_chunks, start=1):
                chunk_id = stable_id(source_id, page, chunk_number, content)
                if chunk_id in seen_ids:
                    raise ValueError(f"duplicate chunk ID generated: {chunk_id}")
                seen_ids.add(chunk_id)
                record = {
                    "chunk_id": chunk_id,
                    "domain": "nutrition",
                    "source_id": source_id,
                    "source": row.get("source", source_id),
                    "title": row.get("title", config["title"]),
                    "page": page,
                    "chunk_number": chunk_number,
                    "knowledge_type": classify_chunk(content, source_id, page),
                    "content": content,
                    "review_required": bool(flags),
                    "review_flags": flags,
                }
                candidate_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunk_count[source_id] += 1
                if flags:
                    review_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    review_count[source_id] += 1

    audit = {
        "input": str(input_path),
        "output_directory": str(output_dir),
        "chunking_parameters": {
            "target_chars": TARGET_CHARS,
            "max_chars": MAX_CHARS,
            "min_chars": MIN_CHARS,
            "overlap": 0,
            "note": "Semantic units are preserved; no automatic overlap.",
        },
        "sources": {
            source_id: {
                "pages_processed": page_count[source_id],
                "source_text_chars": source_chars[source_id],
                "chunks": chunk_count[source_id],
                "review_chunks": review_count[source_id],
            }
            for source_id in SOURCE_CONFIG
        },
        "total_pages": sum(page_count.values()),
        "total_chunks": sum(chunk_count.values()),
        "total_review_chunks": sum(review_count.values()),
        "unique_chunk_ids": len(seen_ids),
    }
    (output_dir / "chunking_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = [
        "NUTRITION CHUNKING CANDIDATE AUDIT",
        f"Pages processed: {audit['total_pages']}",
        f"Candidate chunks: {audit['total_chunks']}",
        f"Chunks requiring review: {audit['total_review_chunks']}",
        f"Unique chunk IDs: {audit['unique_chunk_ids']}",
        "",
        "Candidate data is not approved for ingestion.",
        "Complex tables/charts remain flagged for human review.",
        "ICMR example food charts are reference-only, not fixed meal plans.",
        "ISSN pages 20-25 are excluded.",
        "No database, existing corpus, embeddings, or workout files were modified.",
    ]
    (output_dir / "chunking_audit.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reviewable nutrition source chunks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    audit = build_candidates(args.input, args.output_root)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
