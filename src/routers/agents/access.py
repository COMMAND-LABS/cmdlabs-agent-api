"""
Agent access control.

Single source of truth for "can this account use this agent?" in the
completion API.  This is the completion-api mirror of the same logic in
kalygo3-ai-api.  If the access rules change, **update both**.

Rule
----
An account can access an agent if **either**:
  1. The account **owns** the agent (``agent.account_id == account_id``), OR
  2. The account is a **member** of at least one access group that has been
     **granted** access to the agent.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from src.db.models import Agent, AgentAccessGrant, AccessGroupMember


def can_access_agent(db: Session, account_id: int, agent_id: int) -> bool:
    """
    Return ``True`` if *account_id* is allowed to use *agent_id*.

    Two fast, index-friendly queries at most:
      1. Load the agent and check ownership (PK lookup).
      2. If not owner, check for a group grant + membership (indexed join).
    """
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if agent is None:
        return False

    # Fast path: owner always has access
    if agent.account_id == account_id:
        return True

    # Group-based access: is there any grant for this agent where
    # the caller is a member of the granted group?
    return (
        db.query(AgentAccessGrant.id)
        .join(
            AccessGroupMember,
            AccessGroupMember.access_group_id == AgentAccessGrant.access_group_id,
        )
        .filter(
            AgentAccessGrant.agent_id == agent_id,
            AccessGroupMember.account_id == account_id,
        )
        .first()
    ) is not None


def load_agent_with_access_check(
    db: Session,
    account_id: int,
    agent_id: int,
) -> Optional[Agent]:
    """
    Load and return the agent if *account_id* has access, else ``None``.

    Convenience wrapper so the completion flow can get the Agent object
    and do the access check in one call without running the query twice.
    """
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if agent is None:
        return None

    # Owner short-circuit
    if agent.account_id == account_id:
        return agent

    # Group grant check
    grant_exists = (
        db.query(AgentAccessGrant.id)
        .join(
            AccessGroupMember,
            AccessGroupMember.access_group_id == AgentAccessGrant.access_group_id,
        )
        .filter(
            AgentAccessGrant.agent_id == agent_id,
            AccessGroupMember.account_id == account_id,
        )
        .first()
    ) is not None

    return agent if grant_exists else None
