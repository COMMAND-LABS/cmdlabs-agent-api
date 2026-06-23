"""Base factory for HITL-gated plain-text email tools.

All plain-text email tools (SES, Google OAuth, Google SMTP) share the same
pattern: verify a credential at build time, then queue a PendingToolApproval
at invocation time.  This module captures that shared logic.
"""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.db.models import Credential, PendingToolApproval
from src.routers.credentials.encryption import decrypt_credential_data
from src.tools.exceptions import CredentialError

HITL_SENTINEL_KEY = "__approval_required__"
APPROVAL_TTL_MINUTES = 30


def verify_credential(
    credential_id: int,
    account_id: int,
    db: Session,
    required_fields: list[str],
    provider_label: str,
) -> str:
    """Validate a credential and return its ``from_email`` value.

    Raises ``CredentialError`` if anything is wrong.
    """
    credential = db.query(Credential).filter(
        Credential.id == credential_id,
        Credential.account_id == account_id,
    ).first()

    if not credential:
        raise CredentialError(f"Credential {credential_id} not found or not accessible.")

    try:
        data = decrypt_credential_data(credential.encrypted_data)
    except Exception as exc:
        raise CredentialError(f"Failed to decrypt credential {credential_id}: {exc}")

    missing = [k for k in required_fields if not data.get(k)]
    if missing:
        raise CredentialError(
            f"Credential {credential_id} is missing required {provider_label} fields: {missing}. "
            f"Available keys: {list(data.keys())}"
        )

    return data["from_email"]


class _SendEmailInput(BaseModel):
    to_email: str = Field(description="Recipient email address (e.g. user@example.com)")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Plain-text email body")


async def create_hitl_plain_email_tool(
    *,
    tool_config: dict[str, Any],
    account_id: int,
    db: Session,
    tool_type: str,
    tool_name: str,
    required_credential_fields: list[str],
    provider_label: str,
    default_description: str,
    **kwargs,
) -> StructuredTool:
    """Generic factory for HITL-gated plain-text email tools."""
    credential_id = tool_config.get("credentialId")
    description = tool_config.get("description") or default_description

    if not credential_id:
        raise CredentialError(
            f"Missing required field 'credentialId' in {tool_type} tool configuration"
        )

    credential_account_id = kwargs.get("agent_owner_account_id", account_id)
    from_email = verify_credential(
        credential_id, credential_account_id, db, required_credential_fields, provider_label,
    )

    agent_id: int | None = kwargs.get("agent_id")
    chat_session_id: int | None = kwargs.get("chat_session_id_pk")

    from src.db.database import SessionLocal

    async def send_email_impl(to_email: str, subject: str, body: str) -> str:
        """Queue an email for human approval before sending."""
        expires_at = datetime.now(UTC) + timedelta(minutes=APPROVAL_TTL_MINUTES)

        approval_db: Session = SessionLocal()
        try:
            approval = PendingToolApproval(
                account_id=credential_account_id,
                agent_id=agent_id,
                chat_session_id=chat_session_id,
                tool_type=tool_type,
                status="pending",
                payload={
                    "credential_id": credential_id,
                    "to_email": to_email,
                    "subject": subject,
                    "body": body,
                },
                expires_at=expires_at,
            )
            approval_db.add(approval)
            approval_db.commit()
            approval_db.refresh(approval)
            approval_id = approval.id
        except Exception as exc:
            approval_db.rollback()
            return json.dumps({"success": False, "error": f"Failed to queue email for approval: {exc}"})
        finally:
            approval_db.close()

        return json.dumps({
            HITL_SENTINEL_KEY: True,
            "approval_id": approval_id,
            "tool_type": tool_type,
            "preview": {
                "from_email": from_email,
                "to_email": to_email,
                "subject": subject,
                "body": body,
            },
            "message": (
                f"Email to {to_email} has been queued for human review. "
                "It will be sent only after the user approves it."
            ),
        })

    return StructuredTool(
        func=send_email_impl,
        coroutine=send_email_impl,
        name=tool_name,
        description=description,
        args_schema=_SendEmailInput,
    )
