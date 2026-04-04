"""
Send HTML Email Tool via AWS SES — Human-in-the-Loop (HITL) variant.

The agent writes a complete, production-grade HTML email and passes it
directly as the html_body parameter.  No server-side conversion is applied —
the HTML is stored verbatim and sent exactly as authored.

HITL flow (identical to sendTxtEmailWithSes):
  1. Writes a PendingToolApproval record to the shared database.
  2. Returns a HITL sentinel JSON string so the streaming layer emits a
     ``tool_approval_required`` SSE event to the client.
  3. Returns a human-readable "pending" message to the LLM so the conversation
     can continue naturally while the human reviews the request.
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

    The agent authors a complete, production-grade HTML email and passes it
    directly as html_body.  The HTML is stored verbatim and delivered as-is.

    Required tool_config keys:
        - credentialId: int — ID of the stored AWS SES credential
        - description:  str (optional) — LLM guidance
    """
    credential_id = tool_config.get("credentialId")
    description = (
        tool_config.get("description")
        or (
            "Send a beautifully rendered HTML email to a recipient via AWS SES. "
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

    async def send_html_email_impl(to_email: str, subject: str, html_body: str) -> str:
        """Queue an HTML email for human approval before sending."""
        print(f"\n{'='*60}")
        print(f"[SEND HTML EMAIL TOOL] 📬 HITL: queuing HTML email for approval")
        print(f"[SEND HTML EMAIL TOOL]   To      : {to_email}")
        print(f"[SEND HTML EMAIL TOOL]   Subject : {subject}")
        print(f"[SEND HTML EMAIL TOOL]   HTML    : {len(html_body)} chars")
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
                    "html_body": html_body,
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
                "html_body": html_body,
            },
            "message": (
                f"HTML email to {to_email} has been queued for human review. "
                "It will be sent only after the user approves it."
            ),
        })

    _HTML_EMAIL_DESCRIPTION = """\
The complete HTML source for the email body. Author a self-contained,
production-grade HTML email that renders beautifully across all major email
clients (Gmail, Outlook, Apple Mail, Yahoo Mail).

Standards and best practices to follow:
- Use a single-column, table-based layout (not divs) with a max-width of
  600 px, centred with margin: 0 auto — the universal safe layout for email.
- All CSS MUST be inline (style="...").  External stylesheets and <style>
  blocks in <head> are stripped by most email clients.
- Wrap the entire email in an outer 100%-wide table → inner 600 px table.
- Use web-safe font stacks:
    Arial, Helvetica, sans-serif  — for body copy
    Georgia, 'Times New Roman', serif  — for headlines if desired
- Set explicit width, cellpadding="0", cellspacing="0", and
  border="0" on every <table>.
- Use <td> for padding — never rely on margin on block elements.
- Background colours go on <table> or <td>, not on <body>.
- Images must have alt text, explicit width/height, and display:block to
  avoid phantom gaps.
- Use a preheader <span> immediately after <body> open:
    <span style="display:none;max-height:0;overflow:hidden;">
      One-line preview text shown in inbox list…
    </span>
- Minimum tap-target size for links/buttons: 44 px tall.
- Call-to-action buttons: use a <table> with a coloured <td> containing
  a centred <a> tag — never a <button> element.
- Do NOT use JavaScript, CSS files, SVG, video, or form elements.
- Include a plain-text-friendly fallback; the system sends the HTML as the
  primary part and auto-generates a stripped plain-text alternative.
- Use UTF-8 charset and always set lang="en" on <html>.

Minimal skeleton to follow:
<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f4f4f4;">
  <span style="display:none;max-height:0;overflow:hidden;">Preheader text here.</span>
  <table width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#f4f4f4;">
    <tr><td align="center" style="padding:24px 0;">
      <table width="600" cellpadding="0" cellspacing="0" border="0"
             style="background:#ffffff;border-radius:8px;overflow:hidden;">
        <!-- header -->
        <tr><td style="background:#1a1a2e;padding:24px 32px;">
          <p style="margin:0;font-family:Arial,Helvetica,sans-serif;font-size:22px;
                    font-weight:bold;color:#ffffff;">Your Brand</p>
        </td></tr>
        <!-- body -->
        <tr><td style="padding:32px;font-family:Arial,Helvetica,sans-serif;
                        font-size:16px;line-height:1.6;color:#333333;">
          <p style="margin:0 0 16px;">Hello,</p>
          <!-- content paragraphs here -->
        </td></tr>
        <!-- footer -->
        <tr><td style="padding:16px 32px;background:#f4f4f4;
                        font-family:Arial,Helvetica,sans-serif;
                        font-size:12px;color:#888888;text-align:center;">
          <p style="margin:0;">© 2026 Your Company · Unsubscribe</p>
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
"""

    class SendHtmlEmailInput(BaseModel):
        to_email: str = Field(description="Recipient email address (e.g. user@example.com)")
        subject: str = Field(description="Email subject line — concise and compelling, under 60 characters")
        html_body: str = Field(description=_HTML_EMAIL_DESCRIPTION)

    return StructuredTool(
        func=send_html_email_impl,
        coroutine=send_html_email_impl,
        name="send_html_email_with_ses",
        description=description,
        args_schema=SendHtmlEmailInput,
    )
