from hypothesis import HealthCheck, given, settings, strategies as st

from engine.program import build_program
from evaluation.workout_constraints import EQUIPMENT_SETS, LEVELS, LIMITATION_SETS, check_program
from rag.store import load_exercises
from state.profile import UserProfile

EX = load_exercises()
BY_ID = {e.id: e for e in EX}


@settings(max_examples=40, deadline=None, suppress_health_check=list(HealthCheck))
@given(level=st.sampled_from(LEVELS), equipment=st.sampled_from(EQUIPMENT_SETS),
       limits=st.sampled_from(LIMITATION_SETS), days=st.one_of(st.none(), st.integers(2, 6)),
       goal=st.sampled_from(["general", "muscle", "strength", "fat_loss", "endurance"]),
       minutes=st.sampled_from([30, 45, 60, 90]))
def test_generated_programmes_never_break_hard_constraints(level, equipment, limits, days, goal, minutes):
    profile = UserProfile(name="p", screening="clear", experience=level, equipment=equipment,
                          limitations=limits, goal=goal, session_minutes=minutes)
    result = build_program(EX, profile, days, [goal])
    if result["status"] != "ok":
        return
    bad = {k: v for k, v in check_program(result["program"], profile, days, BY_ID).items() if v}
    assert not bad, bad
