"""
Contract test for the canonical agent access-control rule (agent-api copy).

`src/services/agent_access.py` is byte-identical to ai-api's (enforced by
repo-root check-schemas.sh); this proves the rule behaves identically here.

It also doubles as a SCHEMA SMOKE-TEST: it builds the access tables from
agent-api's own ORM models, so a model referencing a column the real schema no
longer has fails here. Run against a DB the ai-api has migrated (the superrepo
verification flow) to turn it into a real cross-service drift check.

THREE ARMS, AND THE THIRD IS THE ONE THAT CROSSES AN ORG

    own it                                    -> yes
    a grant naming you, in this org           -> yes
    a space you are in, that it was put into  -> yes

The third arm replaced access groups. A group was a set of accounts inside one
org that a grant could name; a space is a set of accounts that may come from
several. What matters here is what did NOT change: being in a space with
somebody still reaches nothing except what was deliberately put in the space.

Needs a disposable Postgres (POSTGRES_TEST_URL or POSTGRES_URL). Skipped if none
is reachable. NEVER point this at production.
"""
import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.database import Base
from src.db.models import Account, Agent, Organization, AccessGrant
from src.db.space_models import Space, SpaceMember, SpaceResource
from src.services.agent_access import (
    can_access_agent,
    get_accessible_agent_ids,
    load_agent_with_access_check,
)

ROOT_ORG_ID = 1

OWNER, GRANTEE, OUTSIDER = 1001, 1002, 1003
SPACE_MEMBER, OTHER_SPACE_MEMBER = 1004, 1005
AGENT_ID, UNSHARED_AGENT_ID = 2001, 2002
SHARED_SPACE, OTHER_SPACE = 3001, 3002
MISSING_AGENT_ID = 999999

_DB_URL = os.environ.get("POSTGRES_TEST_URL") or os.environ.get("POSTGRES_URL", "")
_PROD_HOSTS = ("supabase.co", "neon.tech", "rds.amazonaws.com")
_TABLES = [
    t.__table__
    for t in (Account, Organization, Agent, AccessGrant,
              Space, SpaceMember, SpaceResource)
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
    """One agent reachable three ways, and one reachable only by its owner."""
    for acc_id, email in [
        (OWNER, "owner@example.com"),
        (GRANTEE, "grantee@example.com"),
        (OUTSIDER, "outsider@example.com"),
        (SPACE_MEMBER, "in-the-space@example.com"),
        (OTHER_SPACE_MEMBER, "in-another-space@example.com"),
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

    # Arm 3: the agent put into a space, whose members reach it.
    db.add(Space(id=SHARED_SPACE, name="Shared", owner_account_id=OWNER,
                 owner_org_id=ROOT_ORG_ID))
    db.add(Space(id=OTHER_SPACE, name="Unrelated", owner_account_id=OWNER,
                 owner_org_id=ROOT_ORG_ID))
    db.add(SpaceMember(space_id=SHARED_SPACE, account_id=SPACE_MEMBER,
                       tier_key="member"))
    db.add(SpaceMember(space_id=OTHER_SPACE, account_id=OTHER_SPACE_MEMBER,
                       tier_key="member"))
    db.add(SpaceResource(space_id=SHARED_SPACE, resource_type="agent",
                         resource_id=AGENT_ID, added_by_account_id=OWNER))
    db.flush()
    return db


def test_owner_can_access(seed):
    assert can_access_agent(seed, OWNER, AGENT_ID)


def test_a_named_grantee_can_access(seed):
    assert can_access_agent(seed, GRANTEE, AGENT_ID)


def test_a_member_of_the_space_it_was_shared_into_can_access(seed):
    assert can_access_agent(seed, SPACE_MEMBER, AGENT_ID)


def test_a_member_of_a_different_space_cannot(seed):
    """Being in SOME space reaches nothing. The share names one space."""
    assert not can_access_agent(seed, OTHER_SPACE_MEMBER, AGENT_ID)


def test_the_space_reaches_only_what_was_put_in_it(seed):
    """The sharpest edge in the whole design.

    SPACE_MEMBER reaches AGENT_ID. Its owner also owns UNSHARED_AGENT_ID, in
    the same org, created by the same account. If space membership ever leaked
    into "you may see this person's agents", this is the assertion that fails.
    """
    assert not can_access_agent(seed, SPACE_MEMBER, UNSHARED_AGENT_ID)


def test_outsider_cannot_access(seed):
    assert not can_access_agent(seed, OUTSIDER, AGENT_ID)


def test_missing_agent_is_denied(seed):
    assert not can_access_agent(seed, OWNER, MISSING_AGENT_ID)


def test_get_accessible_agent_ids_is_grants_only(seed):
    """Deliberately narrower than can_access_agent, and it must stay that way.

    This function feeds the LIST queries, which add the space arm themselves
    through org_scope.scoped_resources — beside the org predicate the arm has
    to sit next to. Returning space ids here as well would apply that arm twice
    at two different widths.
    """
    assert get_accessible_agent_ids(seed, GRANTEE) == {AGENT_ID}
    assert get_accessible_agent_ids(seed, SPACE_MEMBER) == set()
    assert get_accessible_agent_ids(seed, OWNER) == set()
    assert get_accessible_agent_ids(seed, OUTSIDER) == set()


def test_load_agent_with_access_check(seed):
    assert load_agent_with_access_check(seed, OWNER, AGENT_ID).id == AGENT_ID
    assert load_agent_with_access_check(seed, GRANTEE, AGENT_ID).id == AGENT_ID
    assert load_agent_with_access_check(seed, SPACE_MEMBER, AGENT_ID).id == AGENT_ID
    assert load_agent_with_access_check(seed, OUTSIDER, AGENT_ID) is None
    assert load_agent_with_access_check(seed, OWNER, MISSING_AGENT_ID) is None
