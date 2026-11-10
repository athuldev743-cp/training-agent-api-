import json
from pathlib import Path

import pytest

from evaluation.progression_cases import run_progression_cases
from engine.models import SetResult, Target
from engine.progression import next_target


def test_progression_matches_all_expected_cases():
    r = run_progression_cases(Path("evaluation/datasets/progression_cases.jsonl"))
    wrong = [c for c in r["cases"] if not c["correct"]]
    assert r["total"] >= 14 and not wrong, wrong


def test_empty_results_rejected():
    with pytest.raises(ValueError):
        next_target(Target(exercise="x", sets=3, reps=10), [])
