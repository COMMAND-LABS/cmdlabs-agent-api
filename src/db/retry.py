"""Transient DB error retry helper.

Shared across agents and swarms routers to handle SSL/connection resets
that occur intermittently with Supabase poolers and Cloud Run cold starts.
"""

import time

from sqlalchemy.exc import OperationalError

_TRANSIENT_PATTERNS = (
    "ssl connection has been closed unexpectedly",
    "server closed the connection unexpectedly",
    "connection reset by peer",
    "could not receive data from server",
)


def is_transient_db_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(p in text for p in _TRANSIENT_PATTERNS)


def db_retry_once(db, label: str, fn):
    """Execute *fn*; on a transient SSL/connection error, rollback, close, and retry once."""
    try:
        return fn()
    except OperationalError as exc:
        if not is_transient_db_error(exc):
            raise
        print(f"[DB RETRY] Transient error during {label}; retrying once...")
        try:
            db.rollback()
        except Exception:
            pass
        db.close()
        time.sleep(0.5)
        return fn()
