"""Context router for chat: decides which facts from the user's own state and the exercise library the model sees.
Pure functions (no model). Only the topics a question needs are included. The user's name, age and weight are never used."""
import re

from engine.program import SPLIT_NAMES, Program
from rag.resolve import resolve_exercise
from rag.store import Exercise
from state.profile import UserProfile

MAX_CONTEXT_CHARS = 2200  # roughly 550 tokens

_PROGRAMME = re.compile(
    r"\b(split|programmes?|programs?|routine|schedule|finisher|weeks?|days?|sessions?|upper|lower|plan|short|long|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?"
    r"|(?:my|first|next|today'?s?)\s+(?:\w+\s+)?workouts?)\b", re.IGNORECASE)
_PROFILE = re.compile(
    r"\b(goals?|level|beginner|intermediate|advanced|experience|equipment|dumbbells?|bench|body-?weight"
    r"|limitations?|injur\w*|avoid\w*|minutes?|profile|bar)\b", re.IGNORECASE)
_HISTORY = re.compile(r"\b(logged?|history|last|previous|recent(?:ly)?)\b", re.IGNORECASE)
_MUSCLE_WORDS = re.compile(
    r"\b(back|chest|core|abs|shoulders?|biceps?|triceps?|legs?|glutes?|quads?|hamstrings?|calves|arms?)\b",
    re.IGNORECASE)


def select_topics(text: str, exercises: list[Exercise]) -> list[str]:
    """Which kinds of facts a question needs. Muscle words only count for the programme when no exercise is named."""
    named = resolve_exercise(text, exercises)["status"] != "none"
    topics = []
    if _PROGRAMME.search(text) or (not named and _MUSCLE_WORDS.search(text)):
        topics.append("programme")
    if _PROFILE.search(text):
        topics.append("profile")
    if _HISTORY.search(text):
        topics.append("history")
    if named:
        topics.append("exercise")
    return topics


def _equipment_text(equipment) -> str:
    if equipment is None:
        return "not set"
    return ", ".join(e.replace("_", " ") for e in equipment) or "bodyweight only"


def profile_facts(profile: UserProfile | None) -> str:
    if profile is None:
        return "USER: no profile saved yet."
    avoid = ", ".join(a.replace("_", " ") for a in profile.limitations) or "nothing"
    return (f"USER: level {profile.experience}; goal {profile.goal.replace('_', ' ')}; "
            f"equipment: {_equipment_text(profile.equipment)}; avoids loading: {avoid}; "
            f"session budget {profile.session_minutes} minutes.")


def locked_pulling(exercises: list[Exercise], equipment) -> list[str]:
    """Equipment the user lacks that pulling exercises (rows, pull-ups, chin-ups) need. Unknown equipment -> nothing."""
    if equipment is None:
        return []
    have = set(equipment) | {"bodyweight"}
    needed: set[str] = set()
    for e in exercises:
        if e.movement == "pull" and not set(e.equipment) <= have:
            needed |= set(e.equipment) - have
    return sorted(needed)


def programme_facts(program: Program | None, exercises: list[Exercise], profile: UserProfile | None,
                    next_session_text: str | None) -> str:
    if program is None:
        return "PROGRAMME: none yet (the user can type /program to build one)."
    days = ", ".join(s.weekday for s in program.weeks[0].sessions)
    equipment = _equipment_text(program.equipment)
    avoid = ", ".join(a.replace("_", " ") for a in program.limitations) or "nothing"
    lines = [f"PROGRAMME: 4-week {SPLIT_NAMES[program.split]} split, {program.days_per_week} days a week ({days}). "
             f"Level {program.level}, goal {program.goal.replace('_', ' ')}, equipment {equipment}, avoiding {avoid}."]
    if profile is not None:
        lines.append(f"Session budget: {profile.session_minutes} minutes.")
    lines.append("Week 1 sessions:")
    for s in program.weeks[0].sessions:
        names = "; ".join(e.name for e in s.exercises)
        lines.append(f"- Day {s.day_number} {s.weekday} {s.label} (about {s.est_minutes:g} min): {names}")
    lines.append("Weeks 3-4 add harder variations and more sets.")
    lines += [f"Note: {n}" for n in program.notes]
    locked = locked_pulling(exercises, program.equipment)
    if locked:
        lines.append("Pulling exercises (rows, pull-ups, chin-ups) you cannot use yet need: "
                     + ", ".join(item.replace("_", " ") for item in locked) + ".")
    if next_session_text:
        lines.append(next_session_text)
    return "\n".join(lines)


def exercise_facts(text: str, exercises: list[Exercise]) -> str:
    res = resolve_exercise(text, exercises)
    chosen = [res["exercise"]] if res["status"] == "ok" else res["candidates"][:3]
    by_id = {e.id: e for e in exercises}
    lines = []
    for e in chosen:
        also = [m for m in e.muscles if m not in e.primary]
        easier = by_id[e.regression].name if e.regression else "none"
        harder = by_id[e.progression].name if e.progression else "none"
        lines.append(f"EXERCISE {e.name}: mainly works {', '.join(e.primary)}"
                     + (f" (also {', '.join(also)})" if also else "")
                     + f"; needs {', '.join(e.equipment).replace('_', ' ')}; level {e.difficulty}; "
                       f"easier version: {easier}; harder version: {harder}.")
    return "\n".join(lines)


def history_facts(recent: list[dict], exercises: list[Exercise]) -> str:
    if not recent:
        return "HISTORY: no workouts logged yet."
    names = {e.id: e.name for e in exercises}
    lines = ["HISTORY (most recent first):"]
    for w in recent[:3]:
        done = ", ".join(names.get(i, i) for i in w["exercises"])
        lines.append(f"- {w['performed_at'][:10]}: {done} ({w['n_sets']} sets)")
    return "\n".join(lines)


def library_facts(exercises: list[Exercise]) -> str:
    """The naive 'send everything' arm: a compact table of the whole exercise library."""
    by_id = {e.id: e for e in exercises}
    lines = ["EXERCISE LIBRARY (name: mainly works; equipment; level; easier / harder):"]
    for e in sorted(exercises, key=lambda x: x.name):
        row = f"- {e.name}: {', '.join(e.primary)}; {', '.join(e.equipment)}; {e.difficulty}"
        if e.regression:
            row += f"; easier {by_id[e.regression].name}"
        if e.progression:
            row += f"; harder {by_id[e.progression].name}"
        lines.append(row)
    return "\n".join(lines)


def build_context(topics: list[str], text: str, exercises: list[Exercise], profile: UserProfile | None = None,
                  program: Program | None = None, next_session_text: str | None = None,
                  recent: list[dict] | None = None, everything: bool = False) -> str:
    if everything:
        topics = ["programme", "profile", "history"]
    parts = []
    if "profile" in topics:
        parts.append(profile_facts(profile))
    if "programme" in topics:
        parts.append(programme_facts(program, exercises, profile, next_session_text))
    if "history" in topics:
        parts.append(history_facts(recent or [], exercises))
    if everything:
        parts.append(library_facts(exercises))
    elif "exercise" in topics:
        parts.append(exercise_facts(text, exercises))
    context = "\n".join(p for p in parts if p)
    if not everything and len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS].rsplit("\n", 1)[0]
    return context