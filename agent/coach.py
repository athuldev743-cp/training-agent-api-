"""Deterministic coaching flow around the stored programme (no model involved):
build and save a programme, show today's workout, finish or skip a session (with adaptation advice
and automatic swaps), reopen a session, and collect reminders."""
from datetime import date

from engine.adaptation import decide
from engine.prescription import LEVELS, Prescription
from engine.program import (
    Program,
    allowed_exercises,
    build_program,
    exercise_families,
    ladder_depth,
    load_tiers,
)
from engine.reminders import Reminder, compute_reminders
from engine.swap import swap_exercise
from engine.swap_checks import check_swap
from rag.store import Exercise
from state.history import HistoryStore
from state.profile import UserProfile
from state.programs import ProgramStore

NO_PROGRAM = "You don't have a routine yet. Type /program and I'll build a 4-week programme from your profile."
NO_ACTIVE = "There is no active programme. Type /program to build one."
FINISHED = "You finished your programme. Type /program to build the next one."


def _fmt_date(iso: str) -> str:
    return date.fromisoformat(iso).strftime("%a %d %b")


def _exercise_lines(session) -> list[str]:
    out = []
    for i, e in enumerate(session.exercises, 1):
        u = " s" if e.unit == "seconds" else ""
        out.append(f"    {i}. {e.name}: {e.sets} sets x {e.aim_low}-{e.aim_high}{u} "
                   f"(range {e.rep_min}-{e.rep_max}{u}), rest {e.rest_seconds} s")
    return out


def format_session(program: Program, row: dict) -> str:
    wk = program.weeks[row["week"] - 1]
    s = wk.sessions[row["day_number"] - 1]
    head = (f"Day {row['day_number']} of week {row['week']} - {s.label}, "
            f"{_fmt_date(row['planned_date'])}, about {s.est_minutes:g} min")
    return "\n".join([head, f"Effort: {wk.effort}"] + _exercise_lines(s))


def swap_level(action: str, level: str) -> str:
    """Which level's rep range a swapped-in exercise starts on. After an easier swap the user starts at
    the beginner range (the old exercise was too hard); after a harder swap, one level down (a harder
    variation needs fewer reps), never below beginner. These are heuristics, easy to change here."""
    if action == "regress":
        return "beginner"
    return LEVELS[max(0, LEVELS.index(level) - 1)]


def start_program(exercises: list[Exercise], profile: UserProfile, programs: ProgramStore,
                  today: date, days: int | None = None, history=None, goals: list[str] | None = None) -> dict:
    if goals is None:  # keep the goals chosen for the last programme while the main goal is unchanged
        prev = programs.latest_program()
        if prev is not None and prev["program"].goals and prev["program"].goals[0] == profile.goal:
            goals = list(prev["program"].goals)
    prev = programs.active_program() or programs.latest_program()
    prev_rows = programs.sessions(prev["id"]) if prev else []
    done_block = prev is not None and bool(prev_rows) and not any(r["status"] == "planned" for r in prev_rows)
    from engine.next_block import accessory_ids
    result = build_program(exercises, profile, days, goals=goals, avoid_ids=accessory_ids(prev["program"]) if done_block else None)
    if result["status"] != "ok":
        return {"status": result["status"], "reply": result["reply"]}
    if history is not None:  # use the last 4 weeks of logs: promote, step back, rebalance
        from engine.history_plan import adjust_program
        from engine.next_block import is_ease
        result["program"], extra = adjust_program(result["program"], profile, history, prev, prev_rows, exercises, today, rebalance=not (done_block and is_ease(prev_rows, history)))
        if done_block:
            from engine.next_block import apply_next_block
            result["program"], _note = apply_next_block(result["program"], profile, prev, prev_rows, history, exercises, today)
            if _note:
                extra = list(extra) + [_note]
        result["reply"] += "".join(f"\nNote: {n}" for n in extra)
    pid = programs.save_program(result["program"], today)
    week1 = [_fmt_date(r["planned_date"]) for r in programs.sessions(pid) if r["week"] == 1]
    reply = (result["reply"] + f"\n\nSaved. Week 1: {', '.join(week1)}. "
             "Type /today for your next workout or /week for the whole week.")
    return {"status": "ok", "reply": reply, "program_id": pid}


def todays_workout(programs: ProgramStore, today: date) -> str:
    active = programs.active_program()
    if active is None:
        return FINISHED if programs.latest_program() else NO_PROGRAM
    row = programs.next_session(active["id"])
    when = date.fromisoformat(row["planned_date"])
    if when == today:
        header = "Today:"
    elif when < today:
        header = f"Overdue (planned for {_fmt_date(row['planned_date'])}). Do it today, or type /skip:"
    else:
        header = "Next workout (not today):"
    return header + "\n" + format_session(active["program"], row)


def week_view(programs: ProgramStore) -> str:
    active = programs.active_program()
    if active is None:
        return FINISHED if programs.latest_program() else NO_PROGRAM
    program = active["program"]
    current = programs.next_session(active["id"])["week"]
    wk = program.weeks[current - 1]
    marks = {"done": "[x]", "skipped": "[-]", "planned": "[ ]"}
    lines = [f"WEEK {current} ({wk.phase}) - effort: {wk.effort}"]
    for r in programs.sessions(active["id"]):
        if r["week"] != current:
            continue
        s = wk.sessions[r["day_number"] - 1]
        lines.append(f"{marks[r['status']]} Day {r['day_number']} - {_fmt_date(r['planned_date'])} - "
                     f"{s.label}, about {s.est_minutes:g} min")
        lines += _exercise_lines(s)
    return "\n".join(lines)


def _substitute(programmed: Exercise, session_ids: set[str], exercises: list[Exercise],
                families: dict, history: HistoryStore, today: date) -> Exercise | None:
    """An exercise from the same ladder, with a shared main muscle, that was logged today instead."""
    for cand in sorted(exercises, key=lambda e: e.name):
        if (cand.id == programmed.id or cand.id in session_ids
                or families[cand.id] != families[programmed.id]
                or not set(cand.primary) & set(programmed.primary)):
            continue
        logged = history.sessions_for(cand.id, limit=1)
        if logged and logged[0]["performed_at"][:10] == today.isoformat():
            return cand
    return None


def _regress_set(old: Exercise, exercises: list[Exercise], families: dict, by_id: dict) -> set[str]:
    """The exercise and everything harder on its ladder with the same equipment and main muscle."""
    depth = ladder_depth(old, by_id)
    return {e.id for e in exercises
            if families[e.id] == families[old.id] and e.equipment == old.equipment
            and set(e.primary) & set(old.primary) and ladder_depth(e, by_id) >= depth}


def finish_session(exercises: list[Exercise], profile: UserProfile, programs: ProgramStore,
                   history: HistoryStore, today: date) -> str:
    """Ticks the earliest unfinished session, advises on what was logged today, and applies swaps
    (an easier or harder variation) to the remaining sessions when the evidence says so.
    Each exercise is judged against the range stored in the programme, so swaps carry their own range."""
    active = programs.active_program()
    if active is None:
        return NO_ACTIVE
    row = programs.next_session(active["id"])
    program = active["program"]
    session = program.weeks[row["week"] - 1].sessions[row["day_number"] - 1]
    programs.mark_done(row["id"], when=f"{today.isoformat()}T12:00:00")

    nxt = programs.next_session(active["id"])
    phase = "foundation" if (nxt is not None and nxt["week"] <= 2) else "build"
    equipment = profile.equipment if profile.equipment is not None else []
    allowed = {e.id for e in allowed_exercises(exercises, profile.experience, equipment,
                                               profile.limitations, phase, load_tiers())}
    by_id = {e.id: e for e in exercises}
    families = exercise_families(exercises)
    session_ids = {i.exercise_id for i in session.exercises}

    advice, swaps, unlogged = [], [], []
    for item in session.exercises:
        logged = history.sessions_for(item.exercise_id, limit=3)
        if logged and logged[0]["performed_at"][:10] == today.isoformat():
            rx = Prescription(template="programme", unit=item.unit, rep_min=item.rep_min,
                              rep_max=item.rep_max, sets_start=item.sets, sets_max=item.sets,
                              rest_seconds=item.rest_seconds)
            d = decide(item.exercise_id, logged, rx, by_id, allowed)
            advice.append(f"  - {item.name}: {d.reason}")
            if d.action in ("regress", "promote") and d.next_exercise_id:
                swaps.append((item.exercise_id, d.next_exercise_id, d.action))
            continue
        sub = _substitute(by_id[item.exercise_id], session_ids, exercises, families, history, today)
        if sub is None:
            unlogged.append(item.name)
            continue
        planned_depth = ladder_depth(by_id[item.exercise_id], by_id)
        done_depth = ladder_depth(sub, by_id)
        if done_depth < planned_depth:
            advice.append(f"  - You did {sub.name} instead of {item.name} (an easier version).")
            if sub.id in allowed:
                swaps.append((item.exercise_id, sub.id, "regress"))
        elif done_depth > planned_depth:
            advice.append(f"  - You did {sub.name} instead of {item.name} (a harder version). "
                          "I haven't changed your plan.")
        else:
            advice.append(f"  - You did {sub.name} instead of {item.name}. I haven't changed your plan.")

    applied, current = [], program
    for old_id, new_id, action in swaps:
        old = by_id[old_id]
        old_ids = _regress_set(old, exercises, families, by_id) if action == "regress" else {old_id}
        problems = check_swap(old_ids, new_id, exercises, equipment, profile.limitations, profile.experience)
        if problems:
            advice.append(f"  - I did not change {old.name} to {by_id[new_id].name}: {problems[0]}.")
            continue
        current, count = swap_exercise(current, old_ids, new_id, swap_level(action, profile.experience),
                                       exercises, (row["week"], row["day_number"]))
        if count:
            applied.append(f"{old.name} -> {by_id[new_id].name}")
    if applied:
        programs.update_program(active["id"], current)

    lines = [f"Marked day {row['day_number']} of week {row['week']} ({session.label}) as done."]
    if advice:
        lines.append("What I noticed from today's log:")
        lines += advice
    if applied:
        lines.append(f"Changed your remaining sessions: {', '.join(applied)}. Type /week to see them.")
    elif advice:
        lines.append("Your remaining sessions are unchanged.")
    if unlogged:
        lines.append("Not logged today: " + ", ".join(unlogged) + ". Log them (for example: "
                     "'I did push-ups 3x8 at RPE 6') so I can adjust your next targets.")
    p = programs.progress(active["id"])
    extra = f", {p['skipped']} skipped" if p["skipped"] else ""
    lines.append(f"Programme progress: {p['done']} of {p['total']} sessions done{extra}.")
    if nxt is None:
        lines.append("That was your last session! Take an easier week, then type /program for the next one.")
    return "\n".join(lines)


def skip_session(programs: ProgramStore) -> str:
    active = programs.active_program()
    if active is None:
        return NO_ACTIVE
    row = programs.next_session(active["id"])
    programs.mark_skipped(row["id"])
    return f"Skipped day {row['day_number']} of week {row['week']}. Type /today to see what is next."


def reopen_last(programs: ProgramStore) -> str:
    latest = programs.latest_program()
    if latest is None:
        return "There is no programme yet."
    row = programs.last_resolved_session(latest["id"])
    if row is None:
        return "Nothing to reopen yet."
    programs.reopen_session(row["id"])
    return f"Reopened day {row['day_number']} of week {row['week']} (it was {row['status']}). Swaps are not undone."


def current_reminders(programs: ProgramStore, today: date) -> list[Reminder]:
    latest = programs.latest_program()
    if latest is None:
        return compute_reminders(today, None, [])
    return compute_reminders(today, latest["program"], programs.sessions(latest["id"]))
