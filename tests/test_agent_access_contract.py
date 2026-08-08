"""
Contract test for the canonical agent access-control rule (agent-api copy).

`src/services/agent_access.py` is byte-identical to ai-api's (enforced by
repo-root check-schemas.sh); this proves the rule behaves identically here.

It also doubles as a SCHEMA SMOKE-TEST: it builds the access tables from
agent-api's own ORM models, so a model referencing a column the real schema no
longer has fails here. Run against a DB the ai-api has migrated (the superrepo
verification flow) to turn it into a real cross-service drift check.

TWO ARMS, NEITHER OF WHICH CROSSES AN ORG

    own it                                    -> yes
    a grant naming you, in this org           -> yes

There was a third — "a space you are in, that it was put into" — and it was the
only one that left the tenant. It replaced access groups: a group was a set of
accounts inside one org that a grant could name; a space was a set of accounts
that could come from several. Spaces were removed to simplify the platform.

Keep this file in step with the ai-api copy, which records what those tests
pinned down for whoever restores the arm.

Needs a disposable Postgres (POSTGRES_TEST_URL or POSTGRES_URL). Skipped if none
is reachable. NEVER point this at production.
"""
import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.database import Base
from src.db.models import Account, Agent, Organization, AccessGrant
from src.services.agent_access import (
    can_access_agent,
    get_accessible_agent_ids,
    load_agent_with_access_check,
)

ROOT_ORG_ID = 1

OWNER, GRANTEE, OUTSIDER = 1001, 1002, 1003
AGENT_ID, UNSHARED_AGENT_ID = 2001, 2002
MISSING_AGENT_ID = 999999

_DB_URL = os.environ.get("POSTGRES_TEST_URL") or os.environ.get("POSTGRES_URL", "")
_PROD_HOSTS = ("supabase.co", "neon.tech", "rds.amazonaws.com")
_TABLES = [
    t.__table__
    for t in (Account, Organization, Agent, AccessGrant)
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
    except Exception as exc:  # unreachable DB -> skip, don't fail the suite
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
    """One agent reachable two ways, and one reachable only by its owner."""
    for acc_id, email in [
        (OWNER, "owner@example.com"),
        (GRANTEE, "grantee@example.com"),
        (OUTSIDER, "outsider@example.com"),
    ]:
        db.add(Account(id=acc_id, email=email))
    # Every agent needs a tenant. This suite is single-org.
    if not db.query(Organization).filter(Organization.id == ROOT_ORG_ID).first():
        db.add(Organization(id=ROOT_ORG_ID, name="CMD LABS"))
        db.flush()
    db.add(Agent(id=AGENT_ID, org_id=ROOT_ORG_ID, account_id=OWNER,
                 name="SOP Agent", config={"data": {}}))
    db.add(Agent(id=UNSHARED_AGENT_ID, org_id=ROOT_ORG_ID, account_id=OWNER,
                 name="Private Agent", config={"data": {}}))

    # Arm 2: a grant naming ONE person.
    db.add(AccessGrant(
        org_id=ROOT_ORG_ID,
        principal_type="account",
        principal_id=GRANTEE,
        resource_type="agent",
        resource_id=AGENT_ID,
        role="use",
    ))

    db.flush()
    return db


def test_owner_can_access(seed):
    assert can_access_agent(seed, OWNER, AGENT_ID)


def test_a_named_grantee_can_access(seed):
    assert can_access_agent(seed, GRANTEE, AGENT_ID)


def test_a_grant_reaches_only_the_agent_it_names(seed):
    """A grant on AGENT_ID says nothing about the owner's other agents.

    Same org, same creator, no grant. This is the surviving half of the
    assertion the space tests made: reaching one of somebody's agents never
    means reaching the rest of them.
    """
    assert not can_access_agent(seed, GRANTEE, UNSHARED_AGENT_ID)


def test_outsider_cannot_access(seed):
    assert not can_access_agent(seed, OUTSIDER, AGENT_ID)


def test_missing_agent_is_denied(seed):
    assert not can_access_agent(seed, OWNER, MISSING_AGENT_ID)


def test_get_accessible_agent_ids_is_grants_only(seed):
    """Deliberately narrower than can_access_agent, and it must stay that way.

    It EXCLUDES owned agents — callers union those separately. It also excluded
    the space arm, because the LIST queries add that themselves through
    org_scope.scoped_resources, beside the org predicate it has to sit next to.
    Returning shared ids here as well would apply that arm twice at two
    different widths; the same reasoning applies to whatever replaces spaces.
    """
    assert get_accessible_agent_ids(seed, GRANTEE) == {AGENT_ID}
    assert get_accessible_agent_ids(seed, OWNER) == set()
    assert get_accessible_agent_ids(seed, OUTSIDER) == set()


def test_load_agent_with_access_check(seed):
    assert load_agent_with_access_check(seed, OWNER, AGENT_ID).id == AGENT_ID
    assert load_agent_with_access_check(seed, GRANTEE, AGENT_ID).id == AGENT_ID
    assert load_agent_with_access_check(seed, OUTSIDER, AGENT_ID) is None
    assert load_agent_with_access_check(seed, OWNER, MISSING_AGENT_ID) is None
