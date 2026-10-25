"""Sync core (Step 11). Works over a sqlite3 connection or state.pg_compat.PgConn. The caller supplies user_id (never the payload).
Idempotent, atomic, ownership-checked. Not covered: merging programme CONTENT edited on two devices (only status is merged);
conflicts are reported, not persisted."""
import re

_UID = re.compile(r"[0-9a-f]{32}")
MAX_ITEMS = 5000
RANK = {"planned": 0, "skipped": 1, "done": 2}
PRANK = {"active": 0, "abandoned": 1, "completed": 2}


def _rows(conn, sql, params=()):
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _uid(v):
    return isinstance(v, str) and _UID.fullmatch(v) is not None


def _int(v, lo, hi):
    return isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


def _num(v, lo, hi):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi


def _ok_set(s):
    return (isinstance(s, dict) and isinstance(s.get("exercise_id"), str) and s["exercise_id"] != ""
            and _int(s.get("set_number"), 1, 1000) and _int(s.get("reps"), 0, 3600) and _num(s.get("weight_kg"), 0, 1000)
            and _num(s.get("rpe"), 1, 10) and s.get("completed") in (0, 1, True, False))


def export_changes(conn, user_id):
    workouts = _rows(conn, "SELECT id, uid, performed_at, duration_min FROM workouts WHERE user_id = ? ORDER BY id", (user_id,))
    for w in workouts:
        w["sets"] = _rows(conn, "SELECT exercise_id, set_number, reps, weight_kg, rpe, completed FROM set_logs WHERE workout_id = ? ORDER BY id", (w.pop("id"),))
    programs = _rows(conn, "SELECT id, uid, created_at, start_date, status, data FROM programs WHERE user_id = ? ORDER BY id", (user_id,))
    for p in programs:
        p["sessions"] = _rows(conn, """SELECT ps.uid AS uid, ps.week AS week, ps.day_number AS day_number, ps.session_number AS session_number,
            ps.planned_date AS planned_date, ps.status AS status, ps.completed_at AS completed_at, w.uid AS workout_uid
            FROM program_sessions ps LEFT JOIN workouts w ON w.id = ps.workout_id WHERE ps.program_id = ? ORDER BY ps.id""", (p.pop("id"),))
    deleted = _rows(conn, "SELECT uid, kind, deleted_at FROM deleted_records WHERE user_id = ?", (user_id,))
    return {"workouts": workouts, "programs": programs, "deleted": deleted}


def apply_changes(conn, user_id, payload):
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    for key in ("workouts", "programs", "deleted"):
        items = payload.get(key, [])
        if not isinstance(items, list) or len(items) > MAX_ITEMS:
            raise ValueError(f"{key} must be a list of at most {MAX_ITEMS} items")
    from engine.program import Program
    rep = {"workouts_added": 0, "programs_added": 0, "sessions_added": 0, "sessions_updated": 0, "deleted_applied": 0,
           "duplicates": 0, "skipped_deleted": 0, "rejected_foreign": 0, "invalid": 0, "conflicts": []}
    with conn:
        for d in payload.get("deleted", []):
            if not (isinstance(d, dict) and _uid(d.get("uid")) and d.get("kind") == "workout" and isinstance(d.get("deleted_at"), str)):
                rep["invalid"] += 1
                continue
            if conn.execute("SELECT 1 FROM deleted_records WHERE uid = ?", (d["uid"],)).fetchone() is None:
                conn.execute("INSERT INTO deleted_records (uid, kind, user_id, deleted_at) VALUES (?, ?, ?, ?)",
                             (d["uid"], "workout", user_id, d["deleted_at"]))
            if conn.execute("DELETE FROM workouts WHERE uid = ? AND user_id = ?", (d["uid"], user_id)).rowcount > 0:
                rep["deleted_applied"] += 1

        for w in payload.get("workouts", []):
            sets = w.get("sets") if isinstance(w, dict) else None
            if not (isinstance(w, dict) and _uid(w.get("uid")) and isinstance(w.get("performed_at"), str)
                    and isinstance(sets, list) and sets and all(_ok_set(s) for s in sets)):
                rep["invalid"] += 1
                continue
            if conn.execute("SELECT 1 FROM deleted_records WHERE uid = ? AND user_id = ?", (w["uid"], user_id)).fetchone():
                rep["skipped_deleted"] += 1
                continue
            ex = conn.execute("SELECT user_id FROM workouts WHERE uid = ?", (w["uid"],)).fetchone()
            if ex is not None:
                rep["duplicates" if ex["user_id"] == user_id else "rejected_foreign"] += 1
                continue
            wid = conn.execute("INSERT INTO workouts (user_id, performed_at, duration_min, uid) VALUES (?, ?, ?, ?)",
                               (user_id, w["performed_at"], w.get("duration_min"), w["uid"])).lastrowid
            for s in sets:
                conn.execute("INSERT INTO set_logs (workout_id, exercise_id, set_number, reps, weight_kg, rpe, completed) VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (wid, s["exercise_id"], s["set_number"], s["reps"], s["weight_kg"], s["rpe"], int(bool(s["completed"]))))
            rep["workouts_added"] += 1

        for p in payload.get("programs", []):
            ok = isinstance(p, dict) and _uid(p.get("uid")) and isinstance(p.get("created_at"), str) and isinstance(p.get("start_date"), str) \
                and p.get("status") in PRANK and isinstance(p.get("data"), str) and isinstance(p.get("sessions"), list)
            if ok:
                try:
                    Program.model_validate_json(p["data"])
                except Exception:
                    ok = False
            if not ok:
                rep["invalid"] += 1
                continue
            ex = conn.execute("SELECT id, user_id, status FROM programs WHERE uid = ?", (p["uid"],)).fetchone()
            if ex is None:
                pid = conn.execute("INSERT INTO programs (user_id, created_at, start_date, status, data, uid) VALUES (?, ?, ?, ?, ?, ?)",
                                   (user_id, p["created_at"], p["start_date"], p["status"], p["data"], p["uid"])).lastrowid
                rep["programs_added"] += 1
            elif ex["user_id"] != user_id:
                rep["rejected_foreign"] += 1
                continue
            else:
                pid = ex["id"]
                if PRANK[p["status"]] > PRANK[ex["status"]]:
                    conn.execute("UPDATE programs SET status = ? WHERE id = ? AND user_id = ?", (p["status"], pid, user_id))
            for s in p["sessions"]:
                if not (isinstance(s, dict) and _uid(s.get("uid")) and _int(s.get("week"), 1, 52) and _int(s.get("day_number"), 1, 14)
                        and _int(s.get("session_number"), 1, 20) and isinstance(s.get("planned_date"), str) and s.get("status") in RANK):
                    rep["invalid"] += 1
                    continue
                cur = conn.execute("""SELECT ps.id AS id, ps.status AS status, ps.completed_at AS completed_at, pr.user_id AS owner
                                      FROM program_sessions ps JOIN programs pr ON pr.id = ps.program_id WHERE ps.uid = ?""", (s["uid"],)).fetchone()
                wuid, wid = s.get("workout_uid"), None
                if _uid(wuid):
                    r = conn.execute("SELECT id FROM workouts WHERE uid = ? AND user_id = ?", (wuid, user_id)).fetchone()
                    wid = r["id"] if r else None
                if cur is None:
                    if conn.execute("SELECT 1 FROM program_sessions WHERE program_id = ? AND week = ? AND day_number = ? AND session_number = ?",
                                    (pid, s["week"], s["day_number"], s["session_number"])).fetchone():
                        rep["invalid"] += 1
                        continue
                    conn.execute("""INSERT INTO program_sessions (program_id, week, day_number, session_number, planned_date, status, workout_id, completed_at, uid)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                 (pid, s["week"], s["day_number"], s["session_number"], s["planned_date"], s["status"], wid, s.get("completed_at"), s["uid"]))
                    rep["sessions_added"] += 1
                elif cur["owner"] != user_id:
                    rep["rejected_foreign"] += 1
                else:
                    incoming, current = (RANK[s["status"]], s.get("completed_at") or ""), (RANK[cur["status"]], cur["completed_at"] or "")
                    if incoming > current:
                        conn.execute("UPDATE program_sessions SET status = ?, completed_at = ?, workout_id = COALESCE(CAST(? AS BIGINT), workout_id) WHERE id = ?",
                                     (s["status"], s.get("completed_at"), wid, cur["id"]))
                        rep["sessions_updated"] += 1
                        if cur["status"] not in ("planned", s["status"]):
                            rep["conflicts"].append({"uid": s["uid"], "kept": s["status"], "dropped": cur["status"]})
                    elif incoming < current and s["status"] not in ("planned", cur["status"]):
                        rep["conflicts"].append({"uid": s["uid"], "kept": cur["status"], "dropped": s["status"]})
            left = conn.execute("SELECT COUNT(*) AS n FROM program_sessions WHERE program_id = ? AND status = 'planned'", (pid,)).fetchone()["n"]
            total = conn.execute("SELECT COUNT(*) AS n FROM program_sessions WHERE program_id = ?", (pid,)).fetchone()["n"]
            if total and left == 0:
                conn.execute("UPDATE programs SET status = 'completed' WHERE id = ? AND user_id = ? AND status = 'active'", (pid, user_id))
    return rep
