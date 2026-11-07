"""Generate deterministic long-text and two-page PDF stress fixtures."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "nutrition"

PARAGRAPH = (
    "Resistance training increases the need to consider total energy intake, "
    "training load, recovery, and the overall dietary pattern. Protein foods "
    "provide amino acids used in muscle protein synthesis and tissue repair. "
    "Carbohydrate availability can support demanding exercise, while dietary "
    "fat contributes essential fatty acids and supports absorption of "
    "fat-soluble vitamins. Hydration planning should account for exercise "
    "duration, intensity, sweat losses, and environmental conditions. "
    "Individual requirements vary, so general guidance should not be treated "
    "as a personalized prescription. "
)


def make_pdf(path: Path, page_texts: list[str]) -> None:
    """Write a tiny standards-compliant text PDF without third-party generators."""
    objects: list[bytes] = []

    def obj(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog_id = obj(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = obj(b"<< /Type /Pages /Kids [5 0 R 7 0 R] /Count 2 >>")
    font_id = obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    content_ids = []
    page_ids = []
    for index, text in enumerate(page_texts):
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode("ascii")
        content_id = obj(
            b"<< /Length " + str(len(stream)).encode("ascii") +
            b" >>\nstream\n" + stream + b"\nendstream"
        )
        page_id = obj(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode("ascii")
        )
        content_ids.append(content_id)
        page_ids.append(page_id)

    # The page tree references the two page objects created above.
    objects[pages_id - 1] = (
        f"<< /Type /Pages /Kids [{page_ids[0]} 0 R {page_ids[1]} 0 R] /Count 2 >>"
    ).encode("ascii")
    del catalog_id, content_ids

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(pdf)


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    long_text = "# Long nutrition stress fixture\n\n" + "\n\n".join(
        f"Section {index + 1}. {PARAGRAPH}" for index in range(80)
    )
    (FIXTURES / "long_nutrition_stress.md").write_text(long_text, encoding="utf-8")
    make_pdf(
        FIXTURES / "two_page_nutrition.pdf",
        [
            "PAGE ONE: Nutrition basics. Protein contributes amino acids for tissue repair.",
            "PAGE TWO: Exercise hydration. Fluid needs vary with duration and conditions.",
        ],
    )
    print("Generated long Markdown document and two-page PDF fixture.")


if __name__ == "__main__":
    main()
