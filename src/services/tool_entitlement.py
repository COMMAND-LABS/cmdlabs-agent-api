"""
Module entitlement for the AGENT RUNTIME.

cmdlabs-api gates its HTTP surface with require_module(): a member whose tier
excludes Contacts gets a 404 from /api/contacts. That closes the front door
only. An agent's tools read the same tables from this service, over their own
sessions, and knew nothing about modules — so the same member could ask an
agent to list their contacts and get them.

This closes that: a tool whose module the caller cannot open is never built, so
it is not in the model's tool list at all. Absent rather than refusing, which
is both cheaper and quieter — the model does not narrate a capability the
caller was not sold.

    effective = organizations.granted_modules  ∩  organization_tiers.modules

kept deliberately identical to cmdlabs-api/src/services/modules.py, including
the owner and platform-staff bypasses. Two services enforcing entitlement
differently is worse than one enforcing it and one not: the gap is harder to
see.

This is the MODULE axis only. It decides which tools exist; org_scope's
tenant_predicate still decides which rows those tools see. Neither substitutes
for the other — a tool that is entitled but unscoped would still be a leak.
"""
import logging

from sqlalchemy.orm import Session

from src.db.models import Organization, OrganizationMember, OrganizationTier

logger = logging.getLogger(__name__)

# Which module each registered tool type belongs to.
#
# A tool type absent from this map is NOT gated. That is deliberate for
# infrastructure-ish tools (raw DB read/write, which are bound to a credential
# the caller already had to hold), and it is why the map lists every registered
# type explicitly rather than only the gated ones — an unlisted type should be
# an oversight you can see, not a silent default.
TOOL_MODULES = {
    "vectorSearch": "knowledge_bases",
    "vectorSearchWithReranking": "knowledge_bases",
    "contactRead": "contacts",
    "contactEventsRead": "contacts",
    "contactEventWrite": "contacts",
    "sendTxtEmailWithSes": "email_campaigns",
    "sendHtmlEmailWithSes": "email_campaigns",
    # Ungated: bound to a credential grant rather than to a product module.
    "dbTableRead": None,
    "dbTableWrite": None,
}


def effective_modules(db: Session, account_id: int, org_id: int) -> set[str]:
    """Module keys `account_id` may open in `org_id`.

    Returns an empty set when the account is not a member of the org — the
    caller then builds no gated tools at all, which is the right failure
    direction for a check that runs outside the request context.
    """
    row = (
        db.query(Organization.granted_modules, OrganizationMember.tier_key,
                 OrganizationMember.is_owner)
        .join(OrganizationMember, OrganizationMember.org_id == Organization.id)
        .filter(Organization.id == org_id,
                OrganizationMember.account_id == account_id)
        .first()
    )
    if row is None:
        logger.warning(
            "[TOOLS] account %s is not a member of org %s — no gated tools",
            account_id, org_id,
        )
        return set()

    ceiling, tier_key, is_owner = row
    ceiling = set(ceiling or ())

    # An owner reaches their org's whole ceiling regardless of their own tier,
    # matching cmdlabs-api. Platform staff likewise bypass the tier but not the
    # ceiling of the org they are acting in.
    if is_owner:
        return ceiling

    tier = (
        db.query(OrganizationTier.modules)
        .filter(OrganizationTier.org_id == org_id,
                OrganizationTier.tier_key == tier_key)
        .scalar()
    )
    return ceiling & set(tier or ())


def allowed_tool_configs(tool_configs: list, granted: set[str]) -> list:
    """Drop tool configs whose module the caller cannot open."""
    kept = []
    for cfg in tool_configs:
        module = TOOL_MODULES.get(cfg.get("type"))
        if module is not None and module not in granted:
            logger.info(
                "[TOOLS] dropping tool %r — %s not enabled for this caller",
                cfg.get("type"), module,
            )
            continue
        kept.append(cfg)
    return kept
