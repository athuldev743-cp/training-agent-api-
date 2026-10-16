"""Database connection and schema migrations, shared by all state stores."""
import os
import re
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 9

# Rows that predate multi-user support, and the default owner for single-user use (CLI, MCP, demos).
LOCAL_USER_ID = "local"
_USER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,63}")


class DatabaseNotReady(RuntimeError):
    """A read-only open found a database file with an incompatible schema version."""


def check_user_id(user_id) -> str:
    """Validates the owner id a store is bound to (it is also used as a database key)."""
    if not isinstance(user_id, str) or not _USER_ID.fullmatch(user_id):
        raise ValueError(
            "user_id must be 1-64 characters: letters, digits and _ . : @ - (starting with a letter or digit)."
        )
    return user_id


def connect(path=":memory:", read_only: bool = False, timeout: float = 5.0) -> sqlite3.Connection:
    if not read_only and os.getenv("STATE_BACKEND") == "postgres":
        from state.pg_compat import connect_pg
        return connect_pg()
    if read_only:
        if str(path) == ":memory:":
            raise ValueError("read_only needs a database file, not :memory:")
        file = Path(path)
        if not file.is_file():
            raise FileNotFoundError("no database file at the given path")
        conn = sqlite3.connect(file.resolve().as_uri() + "?mode=ro", uri=True, timeout=timeout)
        conn.row_factory = sqlite3.Row
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            conn.close()
            raise DatabaseNotReady(
                f"The training database has schema version {version} but this code expects {SCHEMA_VERSION}. "
                "Open the app once so it can update the file, then try again."
            )
        return conn

    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path), timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        migrate(conn)
    except Exception:
        conn.close()
        raise
    return conn


def _program_sessions_ddl(name: str) -> str:
    """The v6 program_sessions schema (one row per session; several sessions may share a day)."""
    return f"""CREATE TABLE {name} (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   program_id INTEGER NOT NULL
                       REFERENCES programs(id) ON DELETE CASCADE,
                   week INTEGER NOT NULL,
                   day_number INTEGER NOT NULL,
                   session_number INTEGER NOT NULL DEFAULT 1,
                   planned_date TEXT NOT NULL,
                   status TEXT NOT NULL,
                   workout_id INTEGER
                       REFERENCES workouts(id) ON DELETE SET NULL,
                   completed_at TEXT,
                   UNIQUE (program_id, week, day_number, session_number)
               )"""


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone() is not None


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(r[1] == column for r in conn.execute(f"PRAGMA table_info({table})"))


_USER_PROFILE_V7 = """CREATE TABLE {name} (
                   user_id TEXT PRIMARY KEY,
                   name TEXT NOT NULL,
                   age INTEGER,
                   height_cm REAL,
                   weight_kg REAL,
                   goal TEXT NOT NULL,
                   experience TEXT NOT NULL,
                   equipment TEXT,
                   session_minutes INTEGER NOT NULL,
                   limitations TEXT,
                   screening TEXT NOT NULL DEFAULT 'not_asked'
               )"""


def _fk_violations(conn: sqlite3.Connection) -> set:
    return {tuple(r) for r in conn.execute("PRAGMA foreign_key_check").fetchall()}


def _check_not_newer(version: int) -> None:
    if version > SCHEMA_VERSION:
        raise DatabaseNotReady(
            f"The training database has schema version {version}, but this code supports "
            f"only version {SCHEMA_VERSION}."
        )


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    _check_not_newer(version)
    if version == SCHEMA_VERSION:
        return

    # Table rebuilds need foreign keys OFF (SQLite ignores this pragma inside a
    # transaction, so it must be set before BEGIN) and are restored afterwards.
    if conn.in_transaction:
        conn.commit()
    fk_was_on = conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    conn.execute("PRAGMA foreign_keys = OFF")

    try:
        # Python's sqlite3 does not open a transaction before DDL, so begin one explicitly.
        # SQLite DDL is transactional: a failure undoes every step. IMMEDIATE takes the write
        # lock now, so two processes opening an old file at once take turns instead of both migrating.
        conn.execute("BEGIN IMMEDIATE")
        # Re-read under the lock: another process may have finished the migration while we waited.
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        _check_not_newer(version)
        if version == SCHEMA_VERSION:
            conn.rollback()
            return
        violations_before = _fk_violations(conn)

        if version < 1:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS user_profile (
                       id INTEGER PRIMARY KEY CHECK (id = 1),
                       name TEXT NOT NULL,
                       age INTEGER,
                       height_cm REAL,
                       weight_kg REAL,
                       goal TEXT NOT NULL,
                       experience TEXT NOT NULL,
                       equipment TEXT,
                       session_minutes INTEGER NOT NULL
                   )"""
            )

        if version < 2:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS workouts (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       performed_at TEXT NOT NULL,
                       duration_min INTEGER
                   )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS set_logs (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       workout_id INTEGER NOT NULL REFERENCES workouts(id) ON DELETE CASCADE,
                       exercise_id TEXT NOT NULL,
                       set_number INTEGER NOT NULL,
                       reps INTEGER NOT NULL,
                       weight_kg REAL NOT NULL,
                       rpe REAL NOT NULL,
                       completed INTEGER NOT NULL
                   )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_set_logs_exercise ON set_logs(exercise_id)"
            )

        if version < 3:
            conn.execute("ALTER TABLE user_profile ADD COLUMN limitations TEXT")
            conn.execute(
                "ALTER TABLE user_profile ADD COLUMN screening TEXT NOT NULL DEFAULT 'not_asked'"
            )

        if version < 4:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS programs (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       created_at TEXT NOT NULL,
                       start_date TEXT NOT NULL,
                       status TEXT NOT NULL,
                       data TEXT NOT NULL
                   )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS program_sessions (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       program_id INTEGER NOT NULL REFERENCES programs(id) ON DELETE CASCADE,
                       week INTEGER NOT NULL,
                       day_number INTEGER NOT NULL,
                       planned_date TEXT NOT NULL,
                       status TEXT NOT NULL,
                       workout_id INTEGER REFERENCES workouts(id) ON DELETE SET NULL,
                       completed_at TEXT,
                       UNIQUE (program_id, week, day_number)
                   )"""
            )

        if version < 5:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS traces (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       ts TEXT NOT NULL,
                       source TEXT NOT NULL,
                       tool TEXT,
                       status TEXT,
                       text TEXT,
                       route_s REAL,
                       total_s REAL,
                       prompt_tokens INTEGER,
                       llm_calls INTEGER NOT NULL DEFAULT 0,
                       error TEXT
                   )"""
            )

        if version < 6:
            # Parent tables must exist, even in minimal legacy databases.
            conn.execute(
                """CREATE TABLE IF NOT EXISTS workouts (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       performed_at TEXT NOT NULL,
                       duration_min INTEGER
                   )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS programs (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       created_at TEXT NOT NULL,
                       start_date TEXT NOT NULL,
                       status TEXT NOT NULL,
                       data TEXT NOT NULL
                   )"""
            )

            has_sessions = conn.execute(
                """SELECT 1 FROM sqlite_master
                   WHERE type = 'table' AND name = 'program_sessions'"""
            ).fetchone() is not None

            if has_sessions:
                # SQLite's documented rebuild: create the new table, copy, drop the
                # old one, then rename the new one into place. Do NOT rename the old
                # table out of the way: SQLite would rewrite other tables' foreign
                # keys (e.g. traces.session_id) to follow it to the doomed name.
                conn.execute(_program_sessions_ddl("program_sessions_new"))
                conn.execute(
                    """INSERT INTO program_sessions_new
                           (id, program_id, week, day_number, session_number,
                            planned_date, status, workout_id, completed_at)
                       SELECT id, program_id, week, day_number, 1,
                              planned_date, status, workout_id, completed_at
                       FROM program_sessions"""
                )
                conn.execute("DROP TABLE program_sessions")
                conn.execute("ALTER TABLE program_sessions_new RENAME TO program_sessions")
            else:
                conn.execute(_program_sessions_ddl("program_sessions"))

            conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_program_sessions_date
                   ON program_sessions(planned_date)"""
            )

        if version < 7:
            # Multi-user: every owned row gets a user_id. Existing rows belong to LOCAL_USER_ID.
            # set_logs and program_sessions are owned through their parent (workouts / programs).
            if _has_table(conn, "user_profile"):
                conn.execute(_USER_PROFILE_V7.format(name="user_profile_new"))
                conn.execute(
                    """INSERT INTO user_profile_new
                           (user_id, name, age, height_cm, weight_kg, goal, experience,
                            equipment, session_minutes, limitations, screening)
                       SELECT ?, name, age, height_cm, weight_kg, goal, experience,
                              equipment, session_minutes, limitations, screening
                       FROM user_profile""",
                    (LOCAL_USER_ID,),
                )
                conn.execute("DROP TABLE user_profile")
                conn.execute("ALTER TABLE user_profile_new RENAME TO user_profile")
            else:
                conn.execute(_USER_PROFILE_V7.format(name="user_profile"))

            for table in ("workouts", "programs", "traces"):
                if _has_table(conn, table) and not _has_column(conn, table, "user_id"):
                    conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN user_id TEXT NOT NULL DEFAULT '{LOCAL_USER_ID}'"
                    )
            if _has_table(conn, "workouts"):
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_workouts_user ON workouts(user_id, performed_at)"
                )
            if _has_table(conn, "programs"):
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_programs_user_status ON programs(user_id, status)"
                )
            if _has_table(conn, "traces"):
                conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_user ON traces(user_id, id)")

        if version < 8:
            # Sync identity (v8): every synced row gets a client-generated uid; integer ids stay device-local.
            for table in ("workouts", "programs", "program_sessions"):
                if _has_table(conn, table):
                    if not _has_column(conn, table, "uid"):
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN uid TEXT")
                    conn.execute(f"UPDATE {table} SET uid = lower(hex(randomblob(16))) WHERE uid IS NULL")
                    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_uid ON {table}(uid)")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS deleted_records (
                       uid TEXT PRIMARY KEY,
                       kind TEXT NOT NULL,
                       user_id TEXT NOT NULL,
                       deleted_at TEXT NOT NULL
                   )"""
            )

        if version < 9:
            # Accounts and refresh tokens (v9)
            conn.execute(
                """CREATE TABLE IF NOT EXISTS accounts (
                       user_id TEXT PRIMARY KEY,
                       email TEXT NOT NULL UNIQUE,
                       pw_hash TEXT NOT NULL,
                       created_at INTEGER NOT NULL,
                       failed INTEGER NOT NULL DEFAULT 0,
                       locked_until INTEGER NOT NULL DEFAULT 0
                   )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS refresh_tokens (
                       token_hash TEXT PRIMARY KEY,
                       user_id TEXT NOT NULL,
                       expires_at INTEGER NOT NULL,
                       revoked INTEGER NOT NULL DEFAULT 0
                   )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id)")

        new_violations = _fk_violations(conn) - violations_before
        if new_violations:
            raise sqlite3.IntegrityError(
                f"Migration would leave {len(new_violations)} foreign-key violation(s); rolled back."
            )

        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if fk_was_on:
            conn.execute("PRAGMA foreign_keys = ON")