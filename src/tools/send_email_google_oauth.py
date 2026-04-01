"""
Send Plain-Text Email Tool via Google Gmail API (OAuth) — tool type: sendTxtEmailWithGoogleOAuth.
Human-in-the-Loop (HITL) variant.

When the agent calls this tool, execution is NOT immediate.  Instead the tool:
  1. Writes a PendingToolApproval record to the shared database.
  2. Returns a HITL sentinel JSON string so the streaming layer can emit a
     ``tool_approval_required`` SSE event to the client.
  3. Returns a human-readable "pending" message to the LLM so the conversation
     can continue naturally while the human reviews the request.

The actual email is sent only after the user clicks **Approve** in the UI,
which calls the approval endpoint in kalygo3-ai-api.

Required credential fields (service_name = GOOGLE_OAUTH):
  - client_id
  - client_secret
  - refresh_token
  - from_email   (the Gmail address to send from)
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
    """Raised when the stored Google OAuth credential is invalid."""


def _verify_google_oauth_credential(
    credential_id: int, account_id: int, db: Session
) -> None:
    """
    Validate that the credential exists and contains the required Google OAuth fields.
    Called at tool-build time so bad configs surface before the first invocation.
    """
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

    required = ["client_id", "client_secret", "refresh_token", "from_email"]
    missing = [k for k in required if not data.get(k)]
    if missing:
        raise CredentialError(
            f"Credential {credential_id} is missing required Google OAuth fields: {missing}. "
            f"Available keys: {list(data.keys())}"
        )


async def create_send_email_google_oauth_tool(
    tool_config: Dict[str, Any],
    account_id: int,
    db: Session,
    auth_token: Optional[str] = None,
    **kwargs,
) -> StructuredTool:
    """
    Create the HITL-gated send-plain-text-email-via-Google-OAuth tool.

    Required tool_config keys (type = sendTxtEmailWithGoogleOAuth):
        - credentialId: int — ID of the stored GOOGLE_OAUTH credential
        - description:  str (optional) — LLM guidance
    """
    credential_id = tool_config.get("credentialId")
    description = (
        tool_config.get("description")
        or (
            "Send a plain-text email to a recipient using Google Gmail (OAuth). "
            "The email will be reviewed by a human before it is delivered."
        )
    )

    if not credential_id:
        raise CredentialError(
            "Missing required field 'credentialId' in sendTxtEmailWithGoogleOAuth tool configuration"
        )

    credential_account_id = kwargs.get("agent_owner_account_id", account_id)
    _verify_google_oauth_credential(credential_id, credential_account_id, db)

    agent_id: Optional[int] = kwargs.get("agent_id")
    chat_session_id: Optional[int] = kwargs.get("chat_session_id_pk")

    from src.db.database import SessionLocal

    print(
        f"[SEND EMAIL GOOGLE OAUTH TOOL] HITL tool 'send_txt_email_with_google_oauth' ready — "
        f"approvals will be stored in pending_tool_approvals "
        f"(credential_id={credential_id}, account_id={credential_account_id})"
    )

    async def send_email_impl(to_email: str, subject: str, body: str) -> str:
        """Queue an email for human approval before sending via Google Gmail OAuth."""
        print(f"\n{'='*60}")
        print(f"[SEND EMAIL GOOGLE OAUTH TOOL] 📬 HITL: queuing email for approval")
        print(f"[SEND EMAIL GOOGLE OAUTH TOOL]   To      : {to_email}")
        print(f"[SEND EMAIL GOOGLE OAUTH TOOL]   Subject : {subject}")
        print(f"[SEND EMAIL GOOGLE OAUTH TOOL]   Body    : {len(body)} chars")
        print(f"{'='*60}\n")

        expires_at = datetime.now(timezone.utc) + timedelta(minutes=APPROVAL_TTL_MINUTES)

        approval_db: Session = SessionLocal()
        try:
            approval = PendingToolApproval(
                account_id=credential_account_id,
                agent_id=agent_id,
                chat_session_id=chat_session_id,
                tool_type="sendTxtEmailWithGoogleOAuth",
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
            print(f"[SEND EMAIL GOOGLE OAUTH TOOL] ✅ PendingToolApproval created — id={approval_id}")
        except Exception as e:
            approval_db.rollback()
            print(f"[SEND EMAIL GOOGLE OAUTH TOOL] ❌ Failed to create approval record: {e}")
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
            "tool_type": "sendTxtEmailWithGoogleOAuth",
            "preview": {
                "to_email": to_email,
                "subject": subject,
                "body": body,
            },
            "message": (
                f"Email to {to_email} has been queued for human review. "
                "It will be sent only after the user approves it."
            ),
        })

    class SendEmailInput(BaseModel):
        to_email: str = Field(description="Recipient email address (e.g. user@example.com)")
        subject: str = Field(description="Email subject line")
        body: str = Field(description="Plain-text email body")

    return StructuredTool(
        func=send_email_impl,
        coroutine=send_email_impl,
        name="send_txt_email_with_google_oauth",
        description=description,
        args_schema=SendEmailInput,
    )
