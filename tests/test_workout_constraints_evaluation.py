from engine.program import build_program
from evaluation.workout_constraints import CATEGORIES, check_program, generate_profiles
from rag.store import load_exercises
from state.profile import UserProfile

EX = load_exercises()
BY_ID = {e.id: e for e in EX}


def _plan(**kw):
    profile = UserProfile(name="t", screening="clear", **kw)
    return profile, build_program(EX, profile, kw.get("days"), None)["program"]


def test_clean_programme_has_no_violations():
    profile, program = _plan(experience="intermediate", equipment=["dumbbells", "bench"])
    assert program is not None
    assert all(not x for x in check_program(program, profile, None, BY_ID).values())


def test_checker_detects_missing_equipment():
    _, program = _plan(experience="intermediate", equipment=["dumbbells", "bench"])
    weaker = UserProfile(name="t", screening="clear", experience="intermediate", equipment=[])
    assert check_program(program, weaker, None, BY_ID)["equipment"]


def test_checker_detects_bad_prescription_and_session_count():
    profile, program = _plan(experience="beginner", equipment=["dumbbells"])
    broken = program.model_copy(deep=True)
    broken.weeks[0].sessions[0].exercises[0].aim_high = 999
    assert check_program(broken, profile, None, BY_ID)["prescription"]
    assert check_program(program, profile, 5, BY_ID)["session_count"]


def test_dataset_generator_is_deterministic_and_has_guardrails():
    a, b = generate_profiles(20), generate_profiles(20)
    assert a == b
    assert sum(1 for r in a if r["expect_status"] != "ok") == 4
    assert set(CATEGORIES)
