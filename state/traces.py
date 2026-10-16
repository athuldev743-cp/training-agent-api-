"""Observability: one row per user turn (route, latency, tokens, outcome), stored on the device."""
from datetime import datetime

from state.db import LOCAL_USER_ID, check_user_id, connect

TEXT_LIMIT = 120  # only the start of a message is kept

_COLUMNS = "id, ts, source, tool, status, text, route_s, total_s, prompt_tokens, llm_calls, error"


class TraceStore:
    def __init__(self, path=":memory:", store_text: bool = True, user_id=LOCAL_USER_ID):
        self.user_id = check_user_id(user_id)
        self.conn = connect(path)
        self.store_text = store_text  # set False to record no message text at all

    def record(self, source: str, tool: str | None = None, status: str | None = None, text: str = "",
               route_s: float | None = None, total_s: float | None = None, prompt_tokens: int | None = None,
               llm_calls: int = 0, error: str | None = None, ts: str | None = None) -> int:
        stored = text[:TEXT_LIMIT] if self.store_text else None
        cur = self.conn.execute(
            """INSERT INTO traces (user_id, ts, source, tool, status, text, route_s, total_s,
                                   prompt_tokens, llm_calls, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (self.user_id, ts or datetime.now().isoformat(timespec="seconds"), source, tool, status, stored,
             route_s, total_s, prompt_tokens, llm_calls, error),
        )
        self.conn.commit()
        return cur.lastrowid

    def rows(self, limit: int | None = None) -> list[dict]:
        sql = f"SELECT {_COLUMNS} FROM traces WHERE user_id = ? ORDER BY id" + (" DESC LIMIT ?" if limit else "")
        cursor = self.conn.execute(sql, (self.user_id, limit) if limit else (self.user_id,))
        result = [dict(r) for r in cursor.fetchall()]
        return list(reversed(result)) if limit else result

    def close(self) -> None:
        self.conn.close()