"""Extract text from supported nutrition source documents."""
from pathlib import Path
from nutrition_ingestion.models import ExtractedPage

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_PDF_PAGES = 2000

class DocumentExtractionError(ValueError):
    """Raised when a source document cannot be safely extracted."""

def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()

def extract_document(path: Path, source_root: Path) -> tuple[list[ExtractedPage], int]:
    """Return non-empty page records and count pages with no extractable text."""
    path = Path(path)
    root = Path(source_root).resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise DocumentExtractionError("source file must be inside the input directory") from exc
    if path.is_symlink() or not resolved.is_file():
        raise DocumentExtractionError(f"not a regular source file: {relative.as_posix()}")
    if resolved.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise DocumentExtractionError(f"unsupported document type: {resolved.suffix}")
    if resolved.stat().st_size > MAX_FILE_BYTES:
        raise DocumentExtractionError(f"source file exceeds {MAX_FILE_BYTES} bytes")

    source_id = relative.as_posix()
    source = relative.as_posix()
    title = resolved.stem.replace("_", " ").replace("-", " ").strip() or resolved.name
    pages: list[ExtractedPage] = []
    empty_pages = 0

    if resolved.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentExtractionError(
                "PDF support is optional; install with: pip install -e '.[ingestion]'"
            ) from exc
        try:
            reader = PdfReader(str(resolved), strict=False)
            if len(reader.pages) > MAX_PDF_PAGES:
                raise DocumentExtractionError(f"PDF exceeds {MAX_PDF_PAGES} pages")
            for page_number, pdf_page in enumerate(reader.pages, start=1):
                text = _clean_text(pdf_page.extract_text())
                if not text:
                    empty_pages += 1
                    continue
                pages.append(ExtractedPage(
                    source_id=source_id, source=source, title=title,
                    page=page_number, text=text,
                ))
        except DocumentExtractionError:
            raise
        except Exception as exc:
            raise DocumentExtractionError(
                f"could not extract PDF {source}: {type(exc).__name__}"
            ) from exc
    else:
        try:
            text = _clean_text(resolved.read_text(encoding="utf-8-sig"))
        except UnicodeDecodeError as exc:
            raise DocumentExtractionError(f"source is not valid UTF-8 text: {source}") from exc
        if text:
            pages.append(ExtractedPage(
                source_id=source_id, source=source, title=title, page=None, text=text,
            ))
        else:
            empty_pages = 1

    if not pages:
        raise DocumentExtractionError(
            f"no extractable text found in {source}; scanned PDFs need OCR before ingestion"
        )
    return pages, empty_pages
