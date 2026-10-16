"""Storage for one user's 4-week programmes and per-session tracking (planned / done / skipped)."""
from datetime import date, datetime

from engine.program import Program
from engine.schedule import schedule_dates
from state.db import LOCAL_USER_ID, check_user_id, connect

_SESSION_COLUMNS = ("ps.id, ps.week, ps.day_number, ps.session_number, ps.planned_date, "
                    "ps.status, ps.workout_id, ps.completed_at")
_OWNED = "ps.program_id = ? AND p.user_id = ?"


class ProgramStore:
    def __init__(self, path=":memory:", read_only=False, user_id=LOCAL_USER_ID):
        self.user_id = check_user_id(user_id)
        self.conn = connect(path, read_only=read_only)

    @staticmethod
    def _as_dict(row) -> dict:
        return {"id": row["id"], "start_date": row["start_date"], "status": row["status"],
                "program": Program.model_validate_json(row["data"])}

    def save_program(self, program: Program, start: date) -> int:
        """Stores a programme and its dated sessions. This user's active programme is marked abandoned."""
        with self.conn:
            self.conn.execute(
                "UPDATE programs SET status = 'abandoned' WHERE status = 'active' AND user_id = ?",
                (self.user_id,),
            )
            cur = self.conn.execute(
                """INSERT INTO programs (user_id, created_at, start_date, status, data, uid)
                   VALUES (?, ?, ?, 'active', ?, lower(hex(randomblob(16))))""",
                (self.user_id, datetime.now().isoformat(timespec="seconds"), start.isoformat(),
                 program.model_dump_json()),
            )
            program_id = cur.lastrowid

            scheduled_dates = {
                (week, day): planned
                for week, day, planned in schedule_dates(program, start)
            }
            for wk in program.weeks:
                for session in wk.sessions:
                    planned = scheduled_dates[(wk.number, session.day_number)]
                    self.conn.execute(
                        """INSERT INTO program_sessions
                               (program_id, week, day_number, session_number,
                                planned_date, status, uid)
                           VALUES (?, ?, ?, ?, ?, 'planned', lower(hex(randomblob(16))))""",
                        (
                            program_id,
                            wk.number,
                            session.day_number,
                            session.session_number,
                            planned.isoformat(),
                        ),
                    )

        return program_id

    def active_program(self) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM programs WHERE status = 'active' AND user_id = ? ORDER BY id DESC LIMIT 1",
            (self.user_id,),
        ).fetchone()
        return self._as_dict(row) if row else None

    def latest_program(self) -> dict | None:
        """The active programme, or the most recently completed one (abandoned ones are ignored)."""
        row = self.conn.execute(
            """SELECT * FROM programs WHERE status IN ('active', 'completed') AND user_id = ?
               ORDER BY id DESC LIMIT 1""",
            (self.user_id,),
        ).fetchone()
        return self._as_dict(row) if row else None

    def sessions(self, program_id: int) -> list[dict]:
        rows = self.conn.execute(
            f"""SELECT {_SESSION_COLUMNS}
                FROM program_sessions ps JOIN programs p ON p.id = ps.program_id
                WHERE {_OWNED}
                ORDER BY ps.week, ps.day_number, ps.session_number""",
            (program_id, self.user_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def next_session(self, program_id: int) -> dict | None:
        return next((r for r in self.sessions(program_id) if r["status"] == "planned"), None)

    def last_resolved_session(self, program_id: int) -> dict | None:
        row = self.conn.execute(
            f"""SELECT {_SESSION_COLUMNS}
                FROM program_sessions ps JOIN programs p ON p.id = ps.program_id
                WHERE {_OWNED} AND ps.status != 'planned'
                ORDER BY ps.week DESC, ps.day_number DESC, ps.session_number DESC LIMIT 1""",
            (program_id, self.user_id),
        ).fetchone()
        return dict(row) if row else None

    def mark_done(self, session_id: int, workout_id: int | None = None, when: str | None = None) -> None:
        self._resolve(session_id, "done", workout_id, when)

    def mark_skipped(self, session_id: int) -> None:
        self._resolve(session_id, "skipped")

    def reopen_session(self, session_id: int) -> None:
        """Puts a done or skipped session back to planned (and reactivates a completed programme)."""
        program_id = self._owned_program_id(session_id)
        with self.conn:
            self.conn.execute(
                """UPDATE program_sessions SET status = 'planned', workout_id = NULL, completed_at = NULL
                   WHERE id = ? AND program_id IN (SELECT id FROM programs WHERE user_id = ?)""",
                (session_id, self.user_id),
            )
            self.conn.execute(
                "UPDATE programs SET status = 'active' WHERE id = ? AND user_id = ? AND status = 'completed'",
                (program_id, self.user_id),
            )

    def update_program(self, program_id: int, program: Program) -> None:
        """Replaces the stored programme data (used when swaps change the remaining sessions)."""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE programs SET data = ? WHERE id = ? AND user_id = ?",
                (program.model_dump_json(), program_id, self.user_id),
            )
        if cur.rowcount == 0:
            raise LookupError(f"No programme with id {program_id}")

    def progress(self, program_id: int) -> dict:
        counts = {"done": 0, "skipped": 0, "planned": 0}
        for r in self.conn.execute(
            """SELECT ps.status, COUNT(*) AS n
               FROM program_sessions ps JOIN programs p ON p.id = ps.program_id
               WHERE ps.program_id = ? AND p.user_id = ? GROUP BY ps.status""",
            (program_id, self.user_id),
        ):
            counts[r["status"]] = r["n"]
        counts["total"] = sum(counts.values())
        return counts

    def _owned_program_id(self, session_id: int) -> int:
        """The programme a session belongs to, only if this user owns it (missing and foreign look the same)."""
        row = self.conn.execute(
            """SELECT ps.program_id FROM program_sessions ps JOIN programs p ON p.id = ps.program_id
               WHERE ps.id = ? AND p.user_id = ?""",
            (session_id, self.user_id),
        ).fetchone()
        if row is None:
            raise LookupError(f"No programme session with id {session_id}")
        return row["program_id"]

    def _resolve(self, session_id: int, status: str, workout_id: int | None = None,
                 when: str | None = None) -> None:
        program_id = self._owned_program_id(session_id)
        completed = (when or datetime.now().isoformat(timespec="seconds")) if status == "done" else None
        with self.conn:
            # One statement checks both owners and writes, so there is no gap between check and update.
            # A workout that is missing and one that belongs to someone else both match nothing here.
            cur = self.conn.execute(
                """UPDATE program_sessions SET status = ?, workout_id = ?, completed_at = ?
                   WHERE id = ?
                     AND program_id IN (SELECT id FROM programs WHERE user_id = ?)
                     AND (? IS NULL
                          OR EXISTS (SELECT 1 FROM workouts w WHERE w.id = ? AND w.user_id = ?))""",
                (status, workout_id, completed, session_id, self.user_id,
                 workout_id, workout_id, self.user_id),
            )
            if cur.rowcount == 0:
                if workout_id is not None:
                    raise LookupError(f"No workout with id {workout_id}")
                raise LookupError(f"No programme session with id {session_id}")
            remaining = self.conn.execute(
                "SELECT COUNT(*) FROM program_sessions WHERE program_id = ? AND status = 'planned'",
                (program_id,),
            ).fetchone()[0]
            if remaining == 0:
                self.conn.execute(
                    "UPDATE programs SET status = 'completed' WHERE id = ? AND user_id = ? AND status = 'active'",
                    (program_id, self.user_id),
                )

    def close(self) -> None:
        self.conn.close()