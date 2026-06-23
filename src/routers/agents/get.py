"""
Get agent details endpoint.

Returns the configuration of an agent the caller owns or has been granted
access to.  Returns 404 when access is denied to avoid leaking existence.
Supports both JWT and API key authentication.
"""
import logging

from fastapi import APIRouter, HTTPException, Request, status

from src.db.models import Account, Agent
from src.deps import auth_dependency, db_dependency
from src.ratelimit import limiter
from src.routers.agents.access import can_access_agent
from src.routers.agents.models import AgentResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{agent_id}", response_model=AgentResponse)
@limiter.limit("60/minute")
async def get_agent(
    agent_id: int,
    db: db_dependency,
    auth: auth_dependency,
    request: Request,
):
    """
    Get the configuration of a specific agent by ID.

    Returns agents the authenticated caller owns **or** has access to via
    an access group.  Returns 404 when the agent does not exist or the
    caller has no access (to avoid leaking existence).

    Accepts JWT cookie or a ``kalygo_``-prefixed API key in the
    ``Authorization: Bearer <key>`` or ``X-API-Key: <key>`` header.
    """
    try:
        account_id = auth["id"]

        account = db.query(Account).filter(Account.id == account_id).first()
        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Account not found",
            )

        agent = db.query(Agent).filter(Agent.id == agent_id).first()

        if not agent or not can_access_agent(db, account_id, agent_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found",
            )

        return AgentResponse(
            id=agent.id,
            name=agent.name,
            config=agent.config,
            is_owner=(agent.account_id == account_id),
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid agent ID: {e!s}",
        ) from e
    except Exception as e:
        logger.error(f"[GET AGENT] Error retrieving agent {agent_id}: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving agent: {e!s}",
        ) from e
