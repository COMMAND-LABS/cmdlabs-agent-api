"""Tests for the structurally-scoped contact_crm tools.

Two layers:
  - Structural (no DB): the security guarantee — no contact_id / account_id
    parameter on any tool. Always runs.
  - Behavior (Postgres test DB, skipped if unavailable): scoping and account
    isolation against a real database via an injected session factory.
"""

import asyncio
import os
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Account, Contact, ContactEvent
from src.tools.contact_crm import (
    create_contact_read_tool,
    create_contact_events_read_tool,
    create_contact_event_write_tool,
)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Structural guarantee (no DB) — the core of the whole design.
# --------------------------------------------------------------------------

def test_no_tool_exposes_contact_or_account_id():
    builders = [
        create_contact_read_tool,
        create_contact_events_read_tool,
        create_contact_event_write_tool,
    ]
    for builder in builders:
        tool = _run(builder({}, account_id=1, contact_id=42))
        fields = set(tool.args_schema.model_fields)
        assert "contact_id" not in fields, f"{tool.name} leaks contact_id"
        assert "account_id" not in fields, f"{tool.name} leaks account_id"


def test_tool_names_are_stable():
    assert _run(create_contact_read_tool({}, 1, contact_id=1)).name == "get_contact"
    assert _run(create_contact_events_read_tool({}, 1, contact_id=1)).name == "list_contact_events"
    assert _run(create_contact_event_write_tool({}, 1, contact_id=1)).name == "add_contact_event"


def test_tools_fail_safe_when_no_contact_bound():
    # Defensive: prepare_agent_context fails closed before this, but the tool
    # must never run unscoped if contact_id is somehow None.
    tool = _run(create_contact_read_tool({}, account_id=1, contact_id=None))
    result = _run(tool.coroutine())
    assert "error" in result


# --------------------------------------------------------------------------
# Behavior (Postgres) — skipped cleanly if the test DB is not reachable.
# --------------------------------------------------------------------------

_PG_URL = os.environ.get(
    "POSTGRES_TEST_URL", "postgresql://test:test@localhost:5432/kalygo_test"
)

try:
    _engine = create_engine(_PG_URL, connect_args={"connect_timeout": 3})
    _conn_ok = True
    with _engine.connect():
        pass
except Exception:  # noqa: BLE001
    _conn_ok = False

pg_required = pytest.mark.skipif(
    not _conn_ok, reason=f"Postgres test DB not reachable at {_PG_URL}"
)


@pytest.fixture()
def pg():
    """Transactional session factory bound to one connection (rolled back)."""
    connection = _engine.connect()
    trans = connection.begin()
    Session = sessionmaker(bind=connection)
    seed = Session()
    try:
        yield Session, seed
    finally:
        seed.close()
        trans.rollback()
        connection.close()


def _seed_contact(session, account_id, email):
    if not session.query(Account).filter(Account.id == account_id).first():
        session.add(Account(id=account_id, email=f"acct{account_id}-{email}"))
        session.flush()
    c = Contact(account_id=account_id, first_name="Rodolfo", last_name="C", email=email)
    session.add(c)
    session.flush()
    return c


@pg_required
def test_get_contact_returns_only_bound_contact(pg):
    Session, seed = pg
    c = _seed_contact(seed, 1, f"{uuid.uuid4()}@x.com")

    tool = _run(create_contact_read_tool({}, account_id=1, contact_id=c.id,
                                         session_factory=Session))
    result = _run(tool.coroutine())
    assert result["id"] == c.id
    assert result["email"] == c.email


@pg_required
def test_get_contact_account_isolation(pg):
    Session, seed = pg
    c = _seed_contact(seed, 1, f"{uuid.uuid4()}@x.com")

    # Same contact id, but a different account in context -> not found.
    tool = _run(create_contact_read_tool({}, account_id=999, contact_id=c.id,
                                         session_factory=Session))
    result = _run(tool.coroutine())
    assert result == {"error": "Contact not found."}


@pg_required
def test_add_event_forces_scope_and_is_listed(pg):
    Session, seed = pg
    c = _seed_contact(seed, 1, f"{uuid.uuid4()}@x.com")

    write = _run(create_contact_event_write_tool({}, account_id=1, contact_id=c.id,
                                                 session_factory=Session))
    out = _run(write.coroutine(event_type="call", title="Intro call",
                               description="Discussed pricing"))
    assert out["success"] is True
    assert out["event"]["event_type"] == "call"

    # The written row carries the forced scope.
    row = seed.query(ContactEvent).filter(ContactEvent.id == out["event"]["id"]).first()
    assert row.contact_id == c.id
    assert row.account_id == 1

    read = _run(create_contact_events_read_tool({}, account_id=1, contact_id=c.id,
                                                session_factory=Session))
    listed = _run(read.coroutine())
    assert any(e["title"] == "Intro call" for e in listed["events"])


@pg_required
def test_list_events_account_isolation(pg):
    Session, seed = pg
    c = _seed_contact(seed, 1, f"{uuid.uuid4()}@x.com")
    seed.add(ContactEvent(contact_id=c.id, account_id=1,
                          event_type="note", title="private"))
    seed.flush()

    read = _run(create_contact_events_read_tool({}, account_id=999, contact_id=c.id,
                                                session_factory=Session))
    listed = _run(read.coroutine())
    assert listed["events"] == []
