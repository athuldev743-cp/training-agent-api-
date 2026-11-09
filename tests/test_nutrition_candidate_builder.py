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
