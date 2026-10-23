"""Plan builder informed by the last 4 weeks of logs. It proposes changes; code validates them.
- an exercise logged at the top of its range on two different days is promoted; one logged below its minimum
  in two sessions in a row is stepped back (engine.adaptation.decide, so one noisy session changes nothing)
- muscles that got clearly fewer sets than the previous programme planned get one extra set where time allows
- days per week are never changed, and the result is discarded if it breaks a hard rule the original passed"""
from datetime import timedelta

from engine.adaptation import decide
from engine.progress import _d, _session, set_rows
from engine.program import MAX_EXERCISES, _exercise_minutes, allowed_exercises, load_templates, load_tiers, prescription_for
from rag.search import DIFFICULTY_RANK
from engine.swap import swap_exercise

WINDOW_DAYS = 28
MIN_WORKOUTS = 2
BEHIND_SHARE = 0.7
MIN_PLANNED_SETS = 6
MAX_SETS = 5


def _in_window(iso, cutoff, today):
    return cutoff <= _d(iso) <= today


def _behind(previous, previous_rows, rows, by_id, cutoff, today):
    if previous is None or len({r["workout_id"] for r in rows}) < MIN_WORKOUTS:
        return []
    planned, done = {}, {}
    for r in previous_rows:
        if not (cutoff <= _d(r["planned_date"]) < today):
            continue
        s = _session(previous["program"], r)
        for it in (s.exercises if s else []):
            for m in by_id[it.exercise_id].primary:
                planned[m] = planned.get(m, 0) + it.sets
    for r in rows:
        if r["completed"] and r["exercise_id"] in by_id:
            for m in by_id[r["exercise_id"]].primary:
                done[m] = done.get(m, 0) + 1
    return sorted(m for m, p in planned.items() if p >= MIN_PLANNED_SETS and done.get(m, 0) < BEHIND_SHARE * p)


def _bump(program, muscles, by_id, templates, budget):
    out = program.model_copy(deep=True)
    changed = False
    for wk in out.weeks:
        for s in wk.sessions:
            for m in muscles:
                it = next((e for e in s.exercises if m in by_id[e.exercise_id].primary and e.sets < MAX_SETS), None)
                if it is None:
                    continue
                it.sets += 1
                minutes = round(sum(_exercise_minutes(prescription_for(e.exercise_id, out.level, templates, out.goal), e.sets)
                                    for e in s.exercises), 1)
                if minutes > budget:
                    it.sets -= 1
                    continue
                s.est_minutes = minutes
                changed = True
    return out, changed


def _violations(program, profile, by_id, tiers):
    """Hard rules checked with deployed modules only (eval/ is not in the bundle)."""
    owned = set(program.equipment or []) | {"bodyweight"}
    ceiling, avoid, cap = DIFFICULTY_RANK[program.level], set(program.limitations), MAX_EXERCISES[program.level]
    out = []
    for wk in program.weeks:
        for s in wk.sessions:
            where = f"week {wk.number} day {s.day_number}"
            ids = [e.exercise_id for e in s.exercises]
            if len(ids) != len(set(ids)):
                out.append(f"{where}: duplicate exercise")
            if len(ids) > cap:
                out.append(f"{where}: more than {cap} exercises")
            if s.est_minutes > profile.session_minutes * 1.1:
                out.append(f"{where}: over the time budget")
            for i in ids:
                ex = by_id[i]
                if not set(ex.equipment) <= owned:
                    out.append(f"{where}: {i} needs equipment the user does not have")
                if set(ex.stresses) & avoid:
                    out.append(f"{where}: {i} loads an area the user avoids")
                if DIFFICULTY_RANK[ex.difficulty] > ceiling:
                    out.append(f"{where}: {i} is above the user's level")
                if program.level == "beginner" and wk.number <= 2 and tiers[i] != "foundation":
                    out.append(f"{where}: {i} is not a foundation exercise")
    return out


def adjust_program(program, profile, history, previous, previous_rows, exercises, today):
    """Returns (programme, notes). The input programme is never modified; history is only read."""
    by_id = {e.id: e for e in exercises}
    cutoff = today - timedelta(days=WINDOW_DAYS)
    rows = [r for r in set_rows(history) if _in_window(r["performed_at"], cutoff, today)]
    if not rows:
        return program, []
    level, goal = program.level, program.goal
    templates, tiers = load_templates(), load_tiers()
    allowed = {e.id for e in allowed_exercises(exercises, level, program.equipment, program.limitations, "build", tiers)}
    plans, seen = [], set()
    for wk in program.weeks:
        for s in wk.sessions:
            for it in s.exercises:
                if it.exercise_id in seen:
                    continue
                seen.add(it.exercise_id)
                logged = [x for x in history.sessions_for(it.exercise_id, 5) if _in_window(x["performed_at"], cutoff, today)]
                if not logged:
                    continue
                d = decide(it.exercise_id, logged, prescription_for(it.exercise_id, level, templates, goal), by_id, allowed)
                if d.action in ("promote", "regress") and d.next_exercise_id:
                    plans.append((it.exercise_id, d.next_exercise_id, d.action))
    adjusted, notes = program, []
    for old, new, action in plans:
        nxt, n = swap_exercise(adjusted, {old}, new, level if action == "promote" else "beginner", exercises, (0, 0))
        if n == 0:
            continue
        adjusted = nxt
        landed = any(e.exercise_id == new for wk in adjusted.weeks for s in wk.sessions for e in s.exercises)
        verb = "moved up" if action == "promote" else "stepped back"
        if landed:
            notes.append(f"From your last 4 weeks of logs: {by_id[old].name} {verb} to {by_id[new].name}.")
        else:
            notes.append(f"From your last 4 weeks of logs: {by_id[old].name} was dropped where {by_id[new].name} would repeat a movement.")
    behind = _behind(previous, previous_rows, rows, by_id, cutoff, today)
    if behind:
        bumped, changed = _bump(adjusted, behind, by_id, templates, profile.session_minutes)
        if changed:
            adjusted = bumped
            notes.append("You did clearly fewer sets than planned for " + ", ".join(m.replace("_", " ") for m in behind) +
                         " over the last 4 weeks, so I added one set there where your time allows. Missed sessions are not squeezed into fewer days.")
    if adjusted is program:
        return program, []
    if set(_violations(adjusted, profile, by_id, tiers)) - set(_violations(program, profile, by_id, tiers)):
        return program, []
    return adjusted.model_copy(update={"notes": list(adjusted.notes) + notes}), notes
