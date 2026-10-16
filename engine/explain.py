"""Deterministic explanation of the user's own programme (no model). The model only routes to this."""
from engine.context import locked_pulling
from engine.program import PRIORITY, SPLIT_NAMES, Program
from rag.store import Exercise
from state.profile import UserProfile


def explain_programme(program: Program, exercises: list[Exercise], profile: UserProfile | None = None) -> str:
    by_id = {e.id: e for e in exercises}
    sessions = program.weeks[0].sessions
    days = ", ".join(s.weekday for s in sessions)
    lines = [f"You follow a {SPLIT_NAMES[program.split]} split, {program.days_per_week} days a week ({days}).",
             "Week 1:"]
    for s in sessions:
        names = ", ".join(e.name for e in s.exercises)
        lines.append(f"- Day {s.day_number} {s.weekday} - {s.label}: {len(s.exercises)} exercises, "
                     f"about {s.est_minutes:g} min ({names}).")
    core_days = [f"day {s.day_number}" for s in sessions
                 if any("core" in by_id[e.exercise_id].primary for e in s.exercises)]
    lines.append("Core work: " + (", ".join(core_days) if core_days else "not in week 1") + ".")
    trained = {m for s in sessions for e in s.exercises for m in by_id[e.exercise_id].primary}
    lines.append("Muscles trained mainly in week 1: " + ", ".join(sorted(trained, key=PRIORITY.index)) + ".")
    locked = locked_pulling(exercises, program.equipment)
    if locked:
        lines.append("Real pulling exercises (rows, pull-ups, chin-ups) are limited because they need: "
                     + ", ".join(item.replace("_", " ") for item in locked) + ".")
    lines += [f"Note: {n}" for n in program.notes if not n.startswith("After week 4")]
    lines.append("Weeks 3-4 add harder variations and more sets.")
    return "\n".join(lines)