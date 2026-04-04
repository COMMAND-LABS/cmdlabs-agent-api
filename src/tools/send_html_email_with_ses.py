"""
Send HTML Email Tool via AWS SES — Human-in-the-Loop (HITL) variant.

The agent passes a plain-text body; each non-empty line is automatically
wrapped in a <p> tag at approval/send time, producing a minimal HTML email
that renders like a text email in any mail client.

HITL flow (identical to sendTxtEmailWithSes):
  1. Writes a PendingToolApproval record to the shared database.
  2. Returns a HITL sentinel JSON string so the streaming layer emits a
     ``tool_approval_required`` SSE event to the client.
  3. Returns a human-readable "pending" message to the LLM so the conversation
     can continue naturally while the human reviews the request.

The HTML conversion happens only after the user clicks Approve, inside the
approval endpoint in kalygo3-ai-api.
"""
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.db.models import Credential, PendingToolApproval
from src.routers.credentials.encryption import decrypt_credential_data

HITL_SENTINEL_KEY = "__approval_required__"
APPROVAL_TTL_MINUTES = 30


class CredentialError(Exception):
    """Raised when the stored AWS SES credential is invalid."""


def _verify_ses_credential(credential_id: int, account_id: int, db: Session) -> str:
    credential = db.query(Credential).filter(
        Credential.id == credential_id,
        Credential.account_id == account_id,
    ).first()

    if not credential:
        raise CredentialError(
            f"Credential with ID {credential_id} not found or not accessible."
        )

    try:
        data = decrypt_credential_data(credential.encrypted_data)
    except Exception as e:
        raise CredentialError(f"Failed to decrypt credential {credential_id}: {e}")

    required = ["aws_access_key_id", "aws_secret_access_key", "aws_region", "from_email"]
    missing = [k for k in required if not data.get(k)]
    if missing:
        raise CredentialError(
            f"Credential {credential_id} is missing required AWS SES fields: {missing}. "
            f"Available keys: {list(data.keys())}"
        )

    return data["from_email"]


async def create_send_html_email_with_ses_tool(
    tool_config: Dict[str, Any],
    account_id: int,
    db: Session,
    auth_token: Optional[str] = None,
    **kwargs,
) -> StructuredTool:
    """
    Create the HITL-gated send-HTML-email tool.

    The agent writes a plain-text body with newlines separating paragraphs.
    Each non-empty line is wrapped in a <p> tag at send time.

    Required tool_config keys:
        - credentialId: int — ID of the stored AWS SES credential
        - description:  str (optional) — LLM guidance
    """
    credential_id = tool_config.get("credentialId")
    description = (
        tool_config.get("description")
        or (
            "Send an HTML email to a recipient. "
            "Write the body as plain text; separate paragraphs with blank lines. "
            "The email will be reviewed by a human before it is delivered."
        )
    )

    if not credential_id:
        raise CredentialError(
            "Missing required field 'credentialId' in sendHtmlEmailWithSes tool configuration"
        )

    credential_account_id = kwargs.get("agent_owner_account_id", account_id)
    from_email = _verify_ses_credential(credential_id, credential_account_id, db)

    agent_id: Optional[int] = kwargs.get("agent_id")
    chat_session_id: Optional[int] = kwargs.get("chat_session_id_pk")

    from src.db.database import SessionLocal

    print(
        f"[SEND HTML EMAIL TOOL] HITL tool 'send_html_email_with_ses' ready — "
        f"approvals will be stored in pending_tool_approvals "
        f"(credential_id={credential_id}, account_id={credential_account_id})"
    )

    async def send_html_email_impl(to_email: str, subject: str, body: str) -> str:
        """Queue an HTML email for human approval before sending."""
        print(f"\n{'='*60}")
        print(f"[SEND HTML EMAIL TOOL] 📬 HITL: queuing HTML email for approval")
        print(f"[SEND HTML EMAIL TOOL]   To      : {to_email}")
        print(f"[SEND HTML EMAIL TOOL]   Subject : {subject}")
        print(f"[SEND HTML EMAIL TOOL]   Body    : {len(body)} chars")
        print(f"{'='*60}\n")

        expires_at = datetime.now(timezone.utc) + timedelta(minutes=APPROVAL_TTL_MINUTES)

        approval_db: Session = SessionLocal()
        try:
            approval = PendingToolApproval(
                account_id=credential_account_id,
                agent_id=agent_id,
                chat_session_id=chat_session_id,
                tool_type="sendHtmlEmailWithSes",
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
            print(f"[SEND HTML EMAIL TOOL] ✅ PendingToolApproval created — id={approval_id}")
        except Exception as e:
            approval_db.rollback()
            print(f"[SEND HTML EMAIL TOOL] ❌ Failed to create approval record: {e}")
            import traceback
            traceback.print_exc()
            return json.dumps({
                "success": False,
                "error": f"Failed to queue email for approval: {e}",
            })
        finally:
            approval_db.close()

        return json.dumps({
            HITL_SENTINEL_KEY: True,
            "approval_id": approval_id,
            "tool_type": "sendHtmlEmailWithSes",
            "preview": {
                "from_email": from_email,
                "to_email": to_email,
                "subject": subject,
                "body": body,
            },
            "message": (
                f"HTML email to {to_email} has been queued for human review. "
                "It will be sent only after the user approves it."
            ),
        })

    class SendHtmlEmailInput(BaseModel):
        to_email: str = Field(description="Recipient email address (e.g. user@example.com)")
        subject: str = Field(description="Email subject line")
        body: str = Field(
            description=(
                "Email body as plain text. Separate paragraphs with newlines — "
                "each non-empty line will be rendered as its own paragraph."
            )
        )

    return StructuredTool(
        func=send_html_email_impl,
        coroutine=send_html_email_impl,
        name="send_html_email_with_ses",
        description=description,
        args_schema=SendHtmlEmailInput,
    )
