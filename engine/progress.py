"""Weekly tracking and next-session targets from logged history. Pure over rows; the API passes the stores in."""
from datetime import date, datetime, timedelta
from statistics import mean

from engine.overload import HARD_RPE, VOLUME_DROP, deload_reason, deload_sets, next_target

LOADABLE = {"dumbbells", "barbell", "cable", "machine"}
MISSED_SHARE = 1 / 3


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _d(iso: str) -> date:
    return datetime.fromisoformat(iso).date()


def set_rows(history) -> list[dict]:
    rows = history.conn.execute(
        """SELECT w.id AS workout_id, w.performed_at, s.exercise_id, s.reps, s.weight_kg, s.rpe, s.completed
           FROM set_logs s JOIN workouts w ON w.id = s.workout_id WHERE w.user_id = ? ORDER BY w.performed_at""",
        (history.user_id,)).fetchall()
    return [{"workout_id": r["workout_id"], "performed_at": r["performed_at"], "exercise_id": r["exercise_id"],
             "reps": r["reps"], "weight_kg": r["weight_kg"], "rpe": r["rpe"], "completed": bool(r["completed"])} for r in rows]


def week_stats(rows: list[dict], start: date) -> dict:
    end = start + timedelta(days=7)
    mine = [r for r in rows if start <= _d(r["performed_at"]) < end]
    done = [r for r in mine if r["completed"]]
    lifts: dict[str, dict] = {}
    for r in mine:
        lifts.setdefault(r["exercise_id"], {"sets": 0, "best_load": 0.0, "best_reps": 0})["sets"] += 1
    for r in done:
        lifts[r["exercise_id"]]["best_load"] = max(lifts[r["exercise_id"]]["best_load"], r["weight_kg"])
    for r in done:
        L = lifts[r["exercise_id"]]
        if r["weight_kg"] == L["best_load"]:
            L["best_reps"] = max(L["best_reps"], r["reps"])
    return {"start": start.isoformat(), "workouts": len({r["workout_id"] for r in mine}), "sets": len(mine),
            "volume": round(sum(r["reps"] * (r["weight_kg"] or 1) for r in done if r["exercise_id"] not in _timed()), 1),
            "avg_rpe": round(mean(r["rpe"] for r in mine), 2) if mine else None, "lifts": lifts}


def beats(cur: dict, prev: dict) -> bool:
    if cur["sets"] > prev["sets"] or cur["best_load"] > prev["best_load"]:
        return True
    return cur["best_load"] == prev["best_load"] and cur["best_reps"] > prev["best_reps"]


def _failed_streak(rows: list[dict]) -> int:
    by: dict[tuple, list] = {}
    for r in rows:
        by.setdefault((r["exercise_id"], r["workout_id"]), []).append(r)
    per_ex: dict[str, list] = {}
    for (ex, wid), sets in sorted(by.items(), key=lambda kv: (kv[1][0]["performed_at"], kv[0][1]), reverse=True):
        bad = any(not s["completed"] for s in sets) or mean(s["rpe"] for s in sets) >= HARD_RPE
        per_ex.setdefault(ex, []).append(bad)
    best = 0
    for flags in per_ex.values():
        n = 0
        for f in flags:
            if not f: break
            n += 1
        best = max(best, n)
    return best


def deload_now(rows: list[dict], today: date) -> str | None:
    """Reasons come from completed weeks only (never the week still in progress)."""
    start = week_start(today)
    weeks = []
    for k in range(1, 5):
        w = week_stats(rows, start - timedelta(days=7 * k))
        if w["sets"]:
            weeks.append({"avg_rpe": w["avg_rpe"], "volume": w["volume"]})
    return deload_reason(weeks, _failed_streak(rows))


def _session(program, row):
    for wk in program.weeks:
        if wk.number == row["week"]:
            for s in wk.sessions:
                if s.day_number == row["day_number"] and s.session_number == row.get("session_number", 1):
                    return s
    return None


def build_targets(program, next_row: dict, history, exercises: list, today: date) -> dict:
    by_id = {e.id: e for e in exercises}
    reason = deload_now(set_rows(history), today)
    session = _session(program, next_row)
    out = []
    for it in (session.exercises if session else []):
        ex = by_id[it.exercise_id]
        t = next_target(it.sets, it.rep_min, it.rep_max, history.sessions_for(it.exercise_id, 5),
                        loadable=bool(set(ex.equipment) & LOADABLE), small_lift=ex.movement == "isolation")
        if reason:
            t["sets"] = deload_sets(it.sets)
            t["reason"] = "Deload week: about 60% of the sets at easier effort. " + t["reason"]
        out.append({"exercise_id": it.exercise_id, "name": it.name, "unit": it.unit, **t})
    return {"week": next_row["week"], "day_number": next_row["day_number"], "label": session.label if session else None,
            "deload": reason, "exercises": out}


def weekly_summary(today: date, program, session_rows: list[dict], rows: list[dict], exercises: list) -> dict:
    by_id = {e.id: e for e in exercises}
    start = week_start(today)
    cur, prev, prev2 = (week_stats(rows, start - timedelta(days=7 * k)) for k in range(3))

    def plan(week_start_date):
        end = week_start_date + timedelta(days=7)
        mine = [r for r in session_rows if week_start_date <= _d(r["planned_date"]) < end]
        sets: dict[str, int] = {}
        for r in mine:
            s = _session(program, r) if program is not None else None
            for it in (s.exercises if s else []):
                for m in by_id[it.exercise_id].primary:
                    sets[m] = sets.get(m, 0) + it.sets
        return {"planned": len(mine), "skipped": sum(1 for r in mine if r["status"] == "skipped"),
                "statuses": {k: sum(1 for r in mine if r["status"] == k) for k in sorted({r["status"] for r in mine})},
                "planned_sets_by_muscle": sets}

    def muscles(week):
        out: dict[str, int] = {}
        for ex, L in week["lifts"].items():
            for m in by_id[ex].primary:
                out[m] = out.get(m, 0) + L["sets"]
        return out

    cur_plan, prev_plan = plan(start), plan(start - timedelta(days=7))
    deload = deload_now(rows, today)
    lifts = [{"exercise_id": ex, "name": by_id[ex].name, "beat_last_week": beats(L, prev["lifts"][ex])}
             for ex, L in cur["lifts"].items() if ex in prev["lifts"]]
    flags = []
    if prev["sets"] and prev2["sets"] and prev["volume"] < VOLUME_DROP * prev2["volume"]:
        flags.append("volume_dropped")
    if prev["sets"] and prev2["sets"] and prev["avg_rpe"] >= HARD_RPE and prev2["avg_rpe"] >= HARD_RPE:
        flags.append("rpe_high_two_weeks")
    if prev_plan["planned"] and (prev_plan["planned"] - prev["workouts"]) / prev_plan["planned"] > MISSED_SHARE:
        flags.append("missed_more_than_a_third_of_sessions")
    return {"week_start": start.isoformat(),
            "this_week": {**{k: v for k, v in cur.items() if k != "lifts"}, **cur_plan, "sets_by_muscle": muscles(cur)},
            "last_week": {**{k: v for k, v in prev.items() if k != "lifts"}, **prev_plan, "sets_by_muscle": muscles(prev)},
            "overload_rule": {"text": "This week must beat last week on reps, load or sets for each main lift, unless a deload is triggered.",
                              "waived_by_deload": bool(deload), "lifts": lifts,
                              "met": True if deload else (all(x["beat_last_week"] for x in lifts) if lifts else None)},
            "deload": deload, "flags": flags}


from functools import lru_cache as _lru_cache


@_lru_cache(maxsize=1)
def _timed():
    """Exercises logged in seconds (holds, conditioning, cardio): they are not volume (reps x load)."""
    from engine.prescription import TEMPLATES, load_templates
    return {i for i, t in load_templates().items() if TEMPLATES[t][0] == "seconds"}
