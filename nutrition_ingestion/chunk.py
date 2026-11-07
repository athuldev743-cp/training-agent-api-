"""Deterministic, page-aware text chunking."""
import hashlib
import re
from nutrition_ingestion.models import ExtractedPage, NutritionChunk

def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _windows(text: str, max_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = hard_end
        if hard_end < len(text):
            floor = start + max(1, max_chars // 2)
            candidates = [
                text.rfind("\n\n", floor, hard_end),
                text.rfind("\n", floor, hard_end),
                text.rfind(". ", floor, hard_end) + 1,
                text.rfind(" ", floor, hard_end),
            ]
            boundary = max(candidates)
            if boundary > floor:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        while next_start < end and text[next_start].isspace():
            next_start += 1
        start = next_start
    return chunks

def chunk_page(page: ExtractedPage, *, max_chars: int = 1200, overlap: int = 200) -> list[NutritionChunk]:
    if max_chars < 100 or max_chars > 4000:
        raise ValueError("max_chars must be between 100 and 4000")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")
    text = _normalize(page.text)
    chunks = []
    for number, piece in enumerate(_windows(text, max_chars, overlap)):
        identity = "\n".join((page.source_id, str(page.page or 0), str(number), piece))
        chunks.append(NutritionChunk(
            chunk_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            source_id=page.source_id, source=page.source, title=page.title,
            page=page.page, chunk_number=number, text=piece,
            char_count=len(piece), token_estimate=max(1, len(piece.split())),
        ))
    return chunks
