"""Secure connection helpers for the Neon PostgreSQL knowledge store."""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.conninfo import make_conninfo


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def get_connection() -> psycopg.Connection:
    """Open a Neon connection with SSL required.

    The caller owns the connection and must close it, preferably using
    a context manager. Database errors are allowed to propagate.
    """
    database_url = os.getenv("NEON_DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "NEON_DATABASE_URL is missing. Configure it in the project .env file."
        )

    connection_string = make_conninfo(database_url, sslmode="require")
    return psycopg.connect(connection_string, connect_timeout=10)


def check_connection() -> dict[str, str]:
    """Check connectivity and return non-sensitive database information."""
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database(), current_user")
            database, user = cursor.fetchone()

    return {"status": "connected", "database": database, "user": user}
