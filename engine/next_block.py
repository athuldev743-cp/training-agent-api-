"""Next 4-week block: choose push / hold / ease from how the finished block went. Pure rules; the model never decides this.
push: at least 85% of sessions done and average effort RPE 8 or lower -> one more set on the main lifts in weeks 3 and 4.
ease: under 60% done, or average effort RPE 9 or higher -> one fewer set everywhere (never below 2).
hold: otherwise. The builder rotates accessories between blocks; the days per week never change.
Thresholds are this project's drafts and need a coach's review."""
from datetime import timedelta
from statistics import mean

from engine.history_plan import _violations
from engine.progress import _d, set_rows
from engine.program import KEEP_ANCHORS, WEEKLY_SET_CAP, _exercise_minutes, load_templates, load_tiers, prescription_for

EASE_BELOW = 0.6
PUSH_AT = 0.85
EASE_RPE = 9.0
PUSH_RPE = 8.0
MIN_SETS = 2
MAX_SETS = 5


def accessory_ids(program):
    return {e.exercise_id for wk in program.weeks for s in wk.sessions for e in s.exercises[KEEP_ANCHORS:]}


def block_stats(prev_rows, rows):
    total = len(prev_rows)
    if not total:
        return None
    dates = [_d(r["planned_date"]) for r in prev_rows]
    lo, hi = min(dates), max(dates) + timedelta(days=3)
    rpe = [r["rpe"] for r in rows if lo <= _d(r["performed_at"]) <= hi]
    done = sum(1 for r in prev_rows if r["status"] == "done")
    return {"total": total, "done": done, "adherence": done / total, "avg_rpe": round(mean(rpe), 2) if rpe else None}


def choose_mode(adherence, avg_rpe):
    if adherence < EASE_BELOW or (avg_rpe is not None and avg_rpe >= EASE_RPE):
        return "ease"
    if adherence >= PUSH_AT and avg_rpe is not None and avg_rpe <= PUSH_RPE:
        return "push"
    return "hold"


def _minutes(session, program, templates):
    return round(sum(_exercise_minutes(prescription_for(e.exercise_id, program.level, templates, (session.goal or program.goal)), e.sets)
                     for e in session.exercises), 1)


def _muscle_sets(week, by_id):
    t = {}
    for s in week.sessions:
        for e in s.exercises:
            if by_id[e.exercise_id].movement != "conditioning":
                for m in by_id[e.exercise_id].primary:
                    t[m] = t.get(m, 0) + e.sets
    return t


def apply_mode(program, mode, profile, by_id):
    out = program.model_copy(deep=True)
    templates = load_templates()
    if mode == "ease":
        for wk in out.weeks:
            for s in wk.sessions:
                for e in s.exercises:
                    e.sets = max(min(e.sets, MIN_SETS), e.sets - 1)
                s.est_minutes = _minutes(s, out, templates)
    elif mode == "push":
        w3, w4 = out.weeks[2], out.weeks[3]
        for s3, s4 in zip(w3.sessions, w4.sessions):
            for e3 in s3.exercises[:KEEP_ANCHORS]:
                e4 = next((e for e in s4.exercises if e.exercise_id == e3.exercise_id), None)
                if e4 is None or e3.sets >= MAX_SETS or e4.sets >= MAX_SETS:
                    continue
                e3.sets += 1
                e4.sets += 1
                fits = (_minutes(s3, out, templates) <= profile.session_minutes and _minutes(s4, out, templates) <= profile.session_minutes
                        and all(_muscle_sets(w, by_id).get(m, 0) <= WEEKLY_SET_CAP for w in (w3, w4) for m in by_id[e3.exercise_id].primary))
                if fits:
                    s3.est_minutes, s4.est_minutes = _minutes(s3, out, templates), _minutes(s4, out, templates)
                else:
                    e3.sets -= 1
                    e4.sets -= 1
    return out


def _note(mode, st, why=""):
    eff = f" at an average effort of RPE {st['avg_rpe']:g}" if st["avg_rpe"] is not None else ""
    did = f"You did {st['done']} of {st['total']} sessions{eff}."
    if mode == "push":
        return f"Next block: pushing a little. {did} The main lifts get one more set in weeks 3 and 4."
    if mode == "ease":
        return f"Next block: easing off to recover. {did} Every exercise has one fewer set (never below 2) and the sessions keep their exercises."
    return f"Next block: holding the load. {did}{why or ' Do at least ' + format(PUSH_AT, '.0%') + ' of the sessions at RPE ' + format(PUSH_RPE, 'g') + ' or lower to progress faster.'}"


def apply_next_block(program, profile, prev, prev_rows, history, exercises, today):
    """Returns (programme, note). Only acts when the previous block is finished (nothing left planned)."""
    if prev is None or not prev_rows or any(r["status"] == "planned" for r in prev_rows):
        return program, None
    st = block_stats(prev_rows, set_rows(history))
    mode = choose_mode(st["adherence"], st["avg_rpe"])
    by_id = {e.id: e for e in exercises}
    new, why = (apply_mode(program, mode, profile, by_id) if mode != "hold" else program), ""
    if mode != "hold":
        tiers = load_tiers()
        if set(_violations(new, profile, by_id, tiers)) - set(_violations(program, profile, by_id, tiers)):
            new, mode = program, "hold"
        elif mode == "push" and new.model_dump() == program.model_dump():
            mode, why = "hold", " There was no room for extra sets: your sessions are near the time limit, a muscle is at its weekly set limit, or the main lifts already have the highest set count."
    note = _note(mode, st, why)
    return new.model_copy(update={"block_mode": mode, "notes": list(new.notes) + [note]}), note


def is_ease(prev_rows, history):
    """True when the finished block's logs say the next block should ease off."""
    st = block_stats(prev_rows, set_rows(history)) if prev_rows else None
    return bool(st) and choose_mode(st["adherence"], st["avg_rpe"]) == "ease"
