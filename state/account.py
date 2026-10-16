"""Account export and deletion (Step 11). Every statement is scoped to the given user_id."""
from state.sync import export_changes


def export_account(conn, user_id):
    data = export_changes(conn, user_id)
    prof = conn.execute("SELECT * FROM user_profile WHERE user_id = ?", (user_id,)).fetchone()
    data["profile"] = dict(prof) if prof else None
    data["traces"] = [dict(r) for r in conn.execute(
        "SELECT ts, source, tool, status, text FROM traces WHERE user_id = ? ORDER BY id", (user_id,)).fetchall()]
    return data


def delete_account(conn, user_id):
    counts = {}
    with conn:
        for table in ("programs", "workouts", "traces", "user_profile", "deleted_records", "refresh_tokens", "accounts"):
            counts[table] = conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,)).rowcount
    return counts
