"""Postgres support for the state stores (Step 11): makes a psycopg connection behave like the sqlite3 connection the
stores use, plus the Postgres schema. Dates stay TEXT and floats are DOUBLE PRECISION so values match SQLite."""
import os
import re
from datetime import datetime

SCHEMA_VERSION = 9


class Row(tuple):
    """A tuple that also supports row['column'] and dict(row), like sqlite3.Row."""
    def __new__(cls, cols, vals):
        obj = super().__new__(cls, vals)
        obj._cols = list(cols)
        return obj

    def keys(self):
        return list(self._cols)

    def __getitem__(self, k):
        if isinstance(k, str):
            return tuple.__getitem__(self, self._cols.index(k))
        return tuple.__getitem__(self, k)


def _row_factory(cursor):
    cols = [d.name for d in cursor.description] if cursor.description else []
    return lambda vals: Row(cols, vals)


DDL = [
    """CREATE TABLE user_profile (user_id TEXT PRIMARY KEY, name TEXT NOT NULL, age INTEGER, height_cm DOUBLE PRECISION,
       weight_kg DOUBLE PRECISION, goal TEXT NOT NULL, experience TEXT NOT NULL, equipment TEXT, session_minutes INTEGER NOT NULL,
       limitations TEXT, screening TEXT NOT NULL DEFAULT 'not_asked')""",
    """CREATE TABLE workouts (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, user_id TEXT NOT NULL DEFAULT 'local',
       performed_at TEXT NOT NULL, duration_min INTEGER)""",
    """CREATE TABLE set_logs (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
       workout_id BIGINT NOT NULL REFERENCES workouts(id) ON DELETE CASCADE, exercise_id TEXT NOT NULL, set_number INTEGER NOT NULL,
       reps INTEGER NOT NULL, weight_kg DOUBLE PRECISION NOT NULL, rpe DOUBLE PRECISION NOT NULL, completed INTEGER NOT NULL)""",
    "CREATE INDEX idx_set_logs_exercise ON set_logs(exercise_id)",
    "CREATE INDEX idx_workouts_user ON workouts(user_id, performed_at)",
    """CREATE TABLE programs (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, user_id TEXT NOT NULL DEFAULT 'local',
       created_at TEXT NOT NULL, start_date TEXT NOT NULL, status TEXT NOT NULL, data TEXT NOT NULL)""",
    "CREATE INDEX idx_programs_user_status ON programs(user_id, status)",
    """CREATE TABLE program_sessions (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
       program_id BIGINT NOT NULL REFERENCES programs(id) ON DELETE CASCADE, week INTEGER NOT NULL, day_number INTEGER NOT NULL,
       session_number INTEGER NOT NULL DEFAULT 1, planned_date TEXT NOT NULL, status TEXT NOT NULL,
       workout_id BIGINT REFERENCES workouts(id) ON DELETE SET NULL, completed_at TEXT,
       UNIQUE (program_id, week, day_number, session_number))""",
    "CREATE INDEX idx_program_sessions_date ON program_sessions(planned_date)",
    """CREATE TABLE traces (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, user_id TEXT NOT NULL DEFAULT 'local',
       ts TEXT NOT NULL, source TEXT NOT NULL, tool TEXT, status TEXT, text TEXT, route_s DOUBLE PRECISION,
       total_s DOUBLE PRECISION, prompt_tokens INTEGER, llm_calls INTEGER NOT NULL DEFAULT 0, error TEXT)""",
    "CREATE INDEX idx_traces_user ON traces(user_id, id)",
    "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)",
    "ALTER TABLE workouts ADD COLUMN uid TEXT", "ALTER TABLE programs ADD COLUMN uid TEXT", "ALTER TABLE program_sessions ADD COLUMN uid TEXT",
    "CREATE UNIQUE INDEX idx_workouts_uid ON workouts(uid)", "CREATE UNIQUE INDEX idx_programs_uid ON programs(uid)", "CREATE UNIQUE INDEX idx_program_sessions_uid ON program_sessions(uid)",
    "CREATE TABLE deleted_records (uid TEXT PRIMARY KEY, kind TEXT NOT NULL, user_id TEXT NOT NULL, deleted_at TEXT NOT NULL)",
    "CREATE TABLE accounts (user_id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, pw_hash TEXT NOT NULL, created_at BIGINT NOT NULL, failed INTEGER NOT NULL DEFAULT 0, locked_until BIGINT NOT NULL DEFAULT 0)",
    "CREATE TABLE refresh_tokens (token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, expires_at BIGINT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0)",
    "CREATE INDEX idx_refresh_user ON refresh_tokens(user_id)",
]
TABLES = "user_profile, workouts, set_logs, programs, program_sessions, traces, deleted_records, accounts, refresh_tokens"

_UPSERT = re.compile(r"INSERT OR REPLACE INTO (\w+)\s*\((.*?)\)\s*VALUES\s*\((.*?)\)\s*$", re.S | re.I)


def translate(sql: str):
    """SQLite-flavoured SQL -> (Postgres SQL, wants_returning_id)."""
    s = sql.strip()
    m = _UPSERT.match(s)
    if m:
        table, cols, vals = m.group(1), [c.strip() for c in m.group(2).split(",")], m.group(3)
        sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "user_id")
        s = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({vals}) ON CONFLICT (user_id) DO UPDATE SET {sets}"
    s = re.sub(r"GROUP_CONCAT\(DISTINCT ([\w.]+)\)", r"string_agg(DISTINCT \1, ',')", s)
    s = re.sub(r"\?\s+IS NULL", "CAST(? AS BIGINT) IS NULL", s)
    s = s.replace("lower(hex(randomblob(16)))", "replace(gen_random_uuid()::text, '-', '')")
    s = s.replace("?", "%s")
    returning = bool(re.match(r"INSERT INTO (workouts|programs|traces)\b", s, re.I))
    if returning:
        s += " RETURNING id"
    return s, returning


class Cur:
    def __init__(self, cur, lastrowid=None):
        self._c, self.lastrowid, self.rowcount = cur, lastrowid, cur.rowcount

    def fetchone(self):
        return self._c.fetchone() if self._c.description else None

    def fetchall(self):
        return self._c.fetchall() if self._c.description else []

    def __iter__(self):
        return iter(self.fetchall())


class PgConn:
    """sqlite3-style connection over a shared autocommit psycopg connection. close() is a no-op: the owner closes `raw`."""
    def __init__(self, raw, owns=False):
        self.raw, self._depth, self.owns = raw, 0, owns

    def execute(self, sql, params=()):
        s, returning = translate(sql)
        cur = self.raw.execute(s, tuple(params))
        last = None
        if returning:
            row = cur.fetchone()
            last = row[0] if row else None
        return Cur(cur, last)

    def __enter__(self):
        if self._depth == 0:
            self.raw.execute("BEGIN")
        self._depth += 1
        return self

    def __exit__(self, et, ev, tb):
        self._depth -= 1
        if self._depth == 0:
            self.raw.execute("ROLLBACK" if et else "COMMIT")
        return False

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        if self.owns:
            self.raw.close()


def open_raw(direct: bool = True):
    """Autocommit psycopg connection. direct=True uses Neon's non-pooled endpoint (session settings such as search_path persist)."""
    import psycopg
    from psycopg.conninfo import make_conninfo
    import rag.neon  # loads .env
    url = os.getenv("NEON_DATABASE_URL")
    if not url:
        raise RuntimeError("NEON_DATABASE_URL is missing.")
    if direct:
        url = url.replace("-pooler", "", 1)
    return psycopg.connect(make_conninfo(url, sslmode="require"), autocommit=True, row_factory=_row_factory, connect_timeout=20)


def apply_schema(raw) -> None:
    for stmt in DDL:
        raw.execute(stmt)
    raw.execute("INSERT INTO schema_migrations (version, applied_at) VALUES (%s, %s)", (SCHEMA_VERSION, datetime.now().isoformat(timespec="seconds")))


import threading

_tls = threading.local()
_all = []
_lock = threading.Lock()


def _open_schema():
    schema = os.getenv("PG_SCHEMA", "app")
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,40}", schema):
        raise ValueError("PG_SCHEMA must be lowercase letters, digits and underscores")
    raw = open_raw(direct=True)
    try:
        raw.execute("SELECT pg_advisory_lock(727)")
        raw.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        raw.execute(f'SET search_path TO "{schema}"')
        if raw.execute("SELECT to_regclass('schema_migrations')").fetchone()[0] is None:
            apply_schema(raw)
        version = raw.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
        raw.execute("SELECT pg_advisory_unlock(727)")
        if version != SCHEMA_VERSION:
            raise RuntimeError(f"database schema version {version} does not match code version {SCHEMA_VERSION}")
    except Exception:
        raw.close()
        raise
    with _lock:
        _all.append(raw)
    return raw, schema


def connect_pg():
    """Postgres connection for the stores. One connection per worker thread, reused across requests."""
    raw, schema = getattr(_tls, "raw", None), getattr(_tls, "schema", None)
    if raw is not None and schema == os.getenv("PG_SCHEMA", "app"):
        try:
            raw.execute("SELECT 1")
            return PgConn(raw)
        except Exception:
            try:
                raw.close()
            except Exception:
                pass
    raw, schema = _open_schema()
    _tls.raw, _tls.schema = raw, schema
    return PgConn(raw)


def close_all():
    with _lock:
        for raw in _all:
            try:
                raw.close()
            except Exception:
                pass
        _all.clear()
    _tls.raw = None
