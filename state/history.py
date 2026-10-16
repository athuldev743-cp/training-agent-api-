"""Workout history: what one user actually did, stored in SQLite."""
from datetime import datetime

from pydantic import BaseModel, Field

from state.db import LOCAL_USER_ID, check_user_id, connect


class SetLog(BaseModel):
    exercise_id: str = Field(min_length=1)
    reps: int = Field(ge=0, le=100)
    weight_kg: float = Field(default=0.0, ge=0)
    rpe: float = Field(ge=1, le=10)
    completed: bool = True


class HistoryStore:
    def __init__(self, path=":memory:", user_id=LOCAL_USER_ID):
        self.user_id = check_user_id(user_id)
        self.conn = connect(path)

    def log_workout(self, sets: list[SetLog], performed_at: str | None = None,
                    duration_min: int | None = None) -> int:
        """Saves a workout and all its sets together, or nothing at all."""
        if not sets:
            raise ValueError("A workout needs at least one set")
        when = performed_at or datetime.now().isoformat(timespec="seconds")
        with self.conn:  # commits on success, rolls back on any error
            cur = self.conn.execute(
                "INSERT INTO workouts (user_id, performed_at, duration_min, uid) VALUES (?, ?, ?, lower(hex(randomblob(16))))",
                (self.user_id, when, duration_min),
            )
            workout_id = cur.lastrowid
            counters: dict[str, int] = {}
            for s in sets:
                counters[s.exercise_id] = counters.get(s.exercise_id, 0) + 1
                self.conn.execute(
                    """INSERT INTO set_logs
                           (workout_id, exercise_id, set_number, reps, weight_kg, rpe, completed)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (workout_id, s.exercise_id, counters[s.exercise_id],
                     s.reps, s.weight_kg, s.rpe, int(s.completed)),
                )
        return workout_id

    def delete_workout(self, workout_id: int) -> bool:
        """Removes this user's workout and (by cascade) all its sets. Returns True if it existed."""
        with self.conn:
            row = self.conn.execute(
                "SELECT uid FROM workouts WHERE id = ? AND user_id = ?", (workout_id, self.user_id)
            ).fetchone()
            cur = self.conn.execute(
                "DELETE FROM workouts WHERE id = ? AND user_id = ?", (workout_id, self.user_id)
            )
            if cur.rowcount > 0 and row is not None:
                self.conn.execute(
                    "INSERT INTO deleted_records (uid, kind, user_id, deleted_at) VALUES (?, 'workout', ?, ?)",
                    (row["uid"], self.user_id, datetime.now().isoformat(timespec="seconds")),
                )
        return cur.rowcount > 0

    def sessions_for(self, exercise_id: str, limit: int = 5) -> list[dict]:
        """The last `limit` workouts that included this exercise, newest first."""
        ids = [
            r["workout_id"]
            for r in self.conn.execute(
                """SELECT w.id AS workout_id FROM workouts w
                   JOIN set_logs s ON s.workout_id = w.id
                   WHERE s.exercise_id = ? AND w.user_id = ?
                   GROUP BY w.id
                   ORDER BY w.performed_at DESC, w.id DESC
                   LIMIT ?""",
                (exercise_id, self.user_id, limit),
            )
        ]
        sessions = []
        for wid in ids:
            rows = self.conn.execute(
                """SELECT w.performed_at, s.exercise_id, s.reps, s.weight_kg, s.rpe, s.completed
                   FROM set_logs s JOIN workouts w ON w.id = s.workout_id
                   WHERE s.workout_id = ? AND s.exercise_id = ? AND w.user_id = ?
                   ORDER BY s.set_number""",
                (wid, exercise_id, self.user_id),
            ).fetchall()
            sessions.append({
                "workout_id": wid,
                "performed_at": rows[0]["performed_at"],
                "sets": [
                    SetLog(exercise_id=r["exercise_id"], reps=r["reps"], weight_kg=r["weight_kg"],
                           rpe=r["rpe"], completed=bool(r["completed"]))
                    for r in rows
                ],
            })
        return sessions

    def recent_workouts(self, limit: int = 5) -> list[dict]:
        rows = self.conn.execute(
            """SELECT w.id, w.performed_at, w.duration_min,
                      COUNT(s.id) AS n_sets,
                      GROUP_CONCAT(DISTINCT s.exercise_id) AS exercises
               FROM workouts w LEFT JOIN set_logs s ON s.workout_id = w.id
               WHERE w.user_id = ?
               GROUP BY w.id
               ORDER BY w.performed_at DESC, w.id DESC
               LIMIT ?""",
            (self.user_id, limit),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "performed_at": r["performed_at"],
                "duration_min": r["duration_min"],
                "n_sets": r["n_sets"],
                "exercises": sorted(r["exercises"].split(",")) if r["exercises"] else [],
            }
            for r in rows
        ]

    def close(self) -> None:
        self.conn.close()