"""
The agent runtime honours module entitlement.

cmdlabs-api gates its HTTP surface with require_module(), so a member whose tier
excludes Contacts gets a 404 from /api/contacts. That closes the front door
only. An agent's tools read the same tables from THIS service, over their own
sessions, and knew nothing about modules — so the same member could ask an agent
to list their contacts and simply get them. The entitlement was a locked door
next to an open window.

Two layers, matching test_contact_crm_tools.py:
  - pure resolution of a tool list against a module set (no DB), and
  - effective_modules against a real org/tier/member (Postgres, skipped cleanly
    when unreachable).

Note what is NOT asserted here: that the tool refuses at call time. A tool the
caller may not use is never BUILT, so it is absent from the model's tool list
entirely. Absent beats refusing — the model cannot narrate, or be talked into
retrying, a capability that was never offered.
"""
import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import (
    Account,
    Organization,
    OrganizationMember,
    OrganizationTier,
)
from src.services.tool_entitlement import (
    TOOL_MODULES,
    allowed_tool_configs,
    effective_modules,
)

# --------------------------------------------------------------------------
# Tool -> module resolution (no DB)
# --------------------------------------------------------------------------

CRM_AGENT = [
    {"type": "contactRead"},
    {"type": "contactEventsRead"},
    {"type": "vectorSearch", "index": "kb"},
    {"type": "dbTableRead"},
]


def test_ungranted_tools_are_dropped():
    kept = {c["type"] for c in allowed_tool_configs(CRM_AGENT, {"knowledge_bases"})}
    assert "vectorSearch" in kept
    assert "contactRead" not in kept, (
        "a caller whose tier excludes Contacts must not get CRM tools")


def test_ungated_tools_survive_an_empty_grant():
    """Raw DB read/write is bound to a credential the caller already had to
    hold, not to a product module — gating it on one would be arbitrary."""
    kept = {c["type"] for c in allowed_tool_configs(CRM_AGENT, set())}
    assert kept == {"dbTableRead"}


def test_every_registered_tool_type_is_classified():
    """An unlisted tool type defaults to ungated, so the map must name them all
    — an oversight should be visible here rather than shipping unguarded."""
    from src.tools import ToolRegistry

    unclassified = set(ToolRegistry.list_types()) - set(TOOL_MODULES)
    assert not unclassified, (
        f"These tool types are not classified in TOOL_MODULES: {sorted(unclassified)}\n"
        "Map each to a module key, or to None with a reason."
    )


# --------------------------------------------------------------------------
# effective_modules against a real org (Postgres)
# --------------------------------------------------------------------------

_PG_URL = os.environ.get(
    "POSTGRES_TEST_URL", "postgresql://test:test@cmdlabs-test-pg:5432/kalygo_test"
)

try:
    _engine = create_engine(_PG_URL, connect_args={"connect_timeout": 3})
    with _engine.connect():
        pass
    _conn_ok = True
except Exception:  # noqa: BLE001
    _conn_ok = False

pg_required = pytest.mark.skipif(
    not _conn_ok, reason=f"Postgres test DB not reachable at {_PG_URL}"
)


@pytest.fixture()
def pg():
    connection = _engine.connect()
    trans = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        trans.rollback()
        connection.close()


def _org(session, slug, ceiling, tier_key, tier_modules, account_id,
         is_owner=False):
    org = Organization(slug=slug, name=slug, data_scope="shared",
                       granted_modules=ceiling, status="active")
    session.add(org)
    session.flush()
    session.add(OrganizationTier(org_id=org.id, tier_key=tier_key,
                                 label=tier_key, modules=tier_modules))
    session.add(Account(id=account_id, email=f"{slug}-{account_id}@t.test",
                        default_org_id=org.id))
    session.flush()
    session.add(OrganizationMember(org_id=org.id, account_id=account_id,
                                   tier_key=tier_key, granted_by="grant",
                                   is_owner=is_owner))
    session.flush()
    return org


@pg_required
def test_effective_modules_is_ceiling_intersect_tier(pg):
    org = _org(pg, "ent-a", ceiling=["contacts", "agents"], tier_key="member",
               tier_modules=["contacts", "deals"], account_id=77001)
    # deals is in the tier but outside the ceiling; agents is the reverse.
    assert effective_modules(pg, 77001, org.id) == {"contacts"}


@pg_required
def test_an_owner_gets_the_whole_ceiling(pg):
    org = _org(pg, "ent-b", ceiling=["contacts", "agents"], tier_key="member",
               tier_modules=[], account_id=77002, is_owner=True)
    assert effective_modules(pg, 77002, org.id) == {"contacts", "agents"}


@pg_required
def test_a_non_member_gets_nothing(pg):
    """Fails closed. This check runs outside the request context, so it cannot
    lean on get_org_context having already refused."""
    org = _org(pg, "ent-c", ceiling=["contacts"], tier_key="member",
               tier_modules=["contacts"], account_id=77003)
    pg.add(Account(id=77004, email="stranger@t.test"))
    pg.flush()
    assert effective_modules(pg, 77004, org.id) == set()


@pg_required
def test_a_tier_without_contacts_gets_no_crm_tools(pg):
    """The end-to-end shape: entitlement resolved from the database, then
    applied to a real agent's tool list."""
    org = _org(pg, "ent-d", ceiling=["agents", "knowledge_bases"],
               tier_key="member", tier_modules=["agents", "knowledge_bases"],
               account_id=77005)
    granted = effective_modules(pg, 77005, org.id)
    kept = {c["type"] for c in allowed_tool_configs(CRM_AGENT, granted)}
    assert "contactRead" not in kept
    assert "vectorSearch" in kept
