from nutrition.inspection.build_nutrition_chunk_candidates import (
    ICMR_EXAMPLE_CHART_PAGES,
    SOURCE_CONFIG,
    _split_oversized_unit,
    classify_chunk,
    make_chunks,
    normalize_text,
    split_into_units,
)


def test_mojibake_is_repaired_without_changing_clean_unicode():
    assert normalize_text("JÃ¤ger et al. â€“ protein Â± 2") == "Jäger et al. – protein ± 2"
    assert normalize_text("café — protein") == "café — protein"


def test_heading_stays_with_its_explanation_when_extraction_has_no_blank_lines():
    units = split_into_units(
        "INTRODUCTION\nDietary diversity supports a balanced eating pattern.\n"
        "Adequate food variety is important for health."
    )
    assert len(units) == 1
    assert units[0].startswith("INTRODUCTION\n")
    assert "Dietary diversity" in units[0]
    assert "Adequate food variety" in units[0]


def test_table_rows_remain_grouped_as_a_single_semantic_unit():
    text = (
        "Recommended Dietary Allowance\n"
        "Age group    Sex    Protein (g/day)\n"
        "18-29 years    Male    54\n"
        "18-29 years    Female    45"
    )
    units = split_into_units(text)
    assert len(units) == 1
    assert "Age group" in units[0]
    assert "18-29 years    Male    54" in units[0]
    assert "18-29 years    Female    45" in units[0]


def test_oversized_table_repeats_header_and_preserves_every_row():
    header = "Age group    Sex    Protein (g/day)"
    rows = [f"Age {i}    Female    {40 + i}" for i in range(1, 90)]
    parts = _split_oversized_unit("\n".join([header, *rows]), limit=220)
    assert len(parts) > 1
    assert all(part.startswith(header) for part in parts)
    for row in rows:
        assert sum(row in part for part in parts) == 1
    assert all(len(part) <= 220 for part in parts)


def test_oversized_prose_is_split_without_losing_sentences():
    sentences = [f"Evidence statement number {i} supports training recovery." for i in range(80)]
    parts = _split_oversized_unit(" ".join(sentences), limit=180)
    assert len(parts) > 1
    reconstructed = " ".join(parts)
    for sentence in sentences:
        assert sentence in reconstructed
    assert all(len(part) <= 180 for part in parts)


def test_chunk_packing_keeps_all_units_and_respects_maximum():
    units = [f"Section {i}. " + ("Evidence supports the recommendation. " * 12) for i in range(12)]
    chunks = make_chunks(units)
    assert len(chunks) > 1
    assert all(len(chunk) <= 1450 for chunk in chunks)
    for unit in units:
        assert sum(unit in chunk for chunk in chunks) == 1


def test_issn_source_excludes_reference_pages():
    allowed = SOURCE_CONFIG["issn_protein_exercise_2017"]["include_pages"]
    assert 1 in allowed and 19 in allowed
    assert 20 not in allowed and 25 not in allowed


def test_icmr_example_charts_are_reference_only_not_meal_plans():
    for page in (130, 135, 146):
        assert page in ICMR_EXAMPLE_CHART_PAGES
        assert classify_chunk("Breakfast, lunch and dinner chart", "icmr_dietary_guidelines_2024", page) == "illustrative_food_chart_reference_only"


def test_rda_ear_content_is_classified_as_nutrient_requirement():
    assert classify_chunk("Age-wise RDA values and units", "icmr_rda_ear_2020", 4) == "nutrient_requirement"


def test_oversized_rda_table_repeats_section_and_column_headers():
    section = "Protein Requirements By Age"
    header = "Age group    Sex    Protein (g/day)"
    rows = [f"Age {i}    Male    {50 + i}" for i in range(1, 70)]
    parts = _split_oversized_unit("\n".join([section, header, *rows]), limit=240)
    assert len(parts) > 1
    assert all(part.startswith(section + "\n" + header) for part in parts)
    for row in rows:
        assert sum(row in part for part in parts) == 1



def test_two_column_layout_inserts_full_width_heading_after_all_content_above_it():
    from nutrition.inspection.build_nutrition_chunk_candidates import block_order

    blocks = [
        {"block_index": 0, "bbox": [10, 10, 190, 30], "text": "left top"},
        {"block_index": 1, "bbox": [10, 60, 190, 80], "text": "left middle"},
        {"block_index": 2, "bbox": [10, 130, 190, 150], "text": "left bottom"},
        {"block_index": 3, "bbox": [300, 15, 480, 35], "text": "right top"},
        {"block_index": 4, "bbox": [300, 55, 480, 75], "text": "right middle"},
        {"block_index": 5, "bbox": [300, 140, 480, 160], "text": "right bottom"},
        {"block_index": 6, "bbox": [10, 100, 480, 120], "text": "FULL WIDTH HEADING"},
    ]
    ordered = block_order(blocks, page_width=500)
    labels = [block["text"] for block in ordered]
    heading_index = labels.index("FULL WIDTH HEADING")
    assert set(labels[:heading_index]) == {
        "left top", "left middle", "right top", "right middle"
    }
    assert set(labels[heading_index + 1:]) == {"left bottom", "right bottom"}
