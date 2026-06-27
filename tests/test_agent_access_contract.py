"""
Contract test for the canonical agent access-control rule (agent-api copy).

`src/services/agent_access.py` is byte-identical to ai-api's (enforced by
repo-root check-schemas.sh); this proves the rule behaves identically here.

It also doubles as a SCHEMA SMOKE-TEST: it builds the access-group tables from
agent-api's own ORM models, so a model that references a column the real schema
no longer has would fail here. Run against a DB the ai-api has migrated (the
superrepo verification flow) to turn it into a real cross-service drift check.

Needs a disposable Postgres (POSTGRES_TEST_URL or POSTGRES_URL). Skipped if none
is reachable. NEVER point this at production.
"""
import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.database import Base
from src.db.models import (
    Account,
    Agent,
    AccessGroup,
    AccessGroupMember,
    AgentAccessGrant,
)
from src.services.agent_access import (
    can_access_agent,
    get_accessible_agent_ids,
    load_agent_with_access_check,
)

OWNER, MEMBER, OUTSIDER, MEMBER_NOGRANT = 1001, 1002, 1003, 1004
AGENT_ID = 2001
GROUP_GRANTED, GROUP_UNGRANTED = 3001, 3002
MISSING_AGENT_ID = 999999

_DB_URL = os.environ.get("POSTGRES_TEST_URL") or os.environ.get("POSTGRES_URL", "")
_PROD_HOSTS = ("supabase.co", "neon.tech", "rds.amazonaws.com")
_TABLES = [
    t.__table__
    for t in (Account, Agent, AccessGroup, AccessGroupMember, AgentAccessGrant)
]


@pytest.fixture(scope="module")
def engine():
    if not _DB_URL or any(h in _DB_URL for h in _PROD_HOSTS):
        pytest.skip("No disposable POSTGRES_URL for the access contract test")
    try:
        eng = create_engine(
            _DB_URL, pool_pre_ping=True, connect_args={"connect_timeout": 5}
        )
        # Schema smoke-test: build just the access tables from the ORM models.
        Base.metadata.create_all(bind=eng, tables=_TABLES)
    except Exception as exc:  # unreachable DB → skip, don't fail the suite
        pytest.skip(f"Test database unavailable: {exc}")
    return eng


@pytest.fixture()
def db(engine):
    """Transactional session rolled back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def seed(db):
    """Owner-owned agent shared with GROUP_GRANTED; a second ungranted group."""
    for acc_id, email in [
        (OWNER, "owner@example.com"),
        (MEMBER, "member@example.com"),
        (OUTSIDER, "outsider@example.com"),
        (MEMBER_NOGRANT, "member-nogrant@example.com"),
    ]:
        db.add(Account(id=acc_id, email=email))
    db.add(Agent(id=AGENT_ID, account_id=OWNER, name="SOP Agent", config={"data": {}}))
    db.add(AccessGroup(id=GROUP_GRANTED, name="Granted", owner_account_id=OWNER))
    db.add(AccessGroup(id=GROUP_UNGRANTED, name="Ungranted", owner_account_id=OWNER))
    db.add(AccessGroupMember(access_group_id=GROUP_GRANTED, account_id=MEMBER))
    db.add(AccessGroupMember(access_group_id=GROUP_UNGRANTED, account_id=MEMBER_NOGRANT))
    db.add(AgentAccessGrant(agent_id=AGENT_ID, access_group_id=GROUP_GRANTED))
    db.flush()
    return db


def test_owner_can_access(seed):
    assert can_access_agent(seed, OWNER, AGENT_ID)


def test_member_of_granted_group_can_access(seed):
    assert can_access_agent(seed, MEMBER, AGENT_ID)


def test_member_of_ungranted_group_cannot_access(seed):
    assert not can_access_agent(seed, MEMBER_NOGRANT, AGENT_ID)


def test_outsider_cannot_access(seed):
    assert not can_access_agent(seed, OUTSIDER, AGENT_ID)


def test_missing_agent_is_denied(seed):
    assert not can_access_agent(seed, OWNER, MISSING_AGENT_ID)


def test_get_accessible_agent_ids(seed):
    # Only group grants count — owner access is NOT included here.
    assert get_accessible_agent_ids(seed, MEMBER) == {AGENT_ID}
    assert get_accessible_agent_ids(seed, OWNER) == set()
    assert get_accessible_agent_ids(seed, OUTSIDER) == set()
    assert get_accessible_agent_ids(seed, MEMBER_NOGRANT) == set()


def test_load_agent_with_access_check(seed):
    assert load_agent_with_access_check(seed, OWNER, AGENT_ID).id == AGENT_ID
    assert load_agent_with_access_check(seed, MEMBER, AGENT_ID).id == AGENT_ID
    assert load_agent_with_access_check(seed, OUTSIDER, AGENT_ID) is None
    assert load_agent_with_access_check(seed, OWNER, MISSING_AGENT_ID) is None
