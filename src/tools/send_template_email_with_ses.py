"""
Send Template Email Tool via AWS SES — Human-in-the-Loop (HITL) variant.

The agent selects an EmailTemplate by ID, provides variable values, and
optionally overrides the recipient address.  The template is rendered
server-side at invocation time ({{token}} substitution), then the rendered
HTML is stored in a PendingToolApproval for human review.

At approval time the kalygo3-ai-api endpoint injects an open-tracking pixel
and sends via SES.

HITL flow:
  1. Writes a PendingToolApproval record.
  2. Returns a HITL sentinel so the streaming layer emits a
     ``tool_approval_required`` SSE event.
  3. Returns a "pending" message to the LLM so the conversation continues.
"""
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.db.models import Credential, PendingToolApproval, EmailTemplate
from src.routers.credentials.encryption import decrypt_credential_data

HITL_SENTINEL_KEY = "__approval_required__"
APPROVAL_TTL_MINUTES = 30


class CredentialError(Exception):
    pass


def _verify_ses_credential(credential_id: int, account_id: int, db: Session) -> str:
    credential = db.query(Credential).filter(
        Credential.id == credential_id,
        Credential.account_id == account_id,
    ).first()
    if not credential:
        raise CredentialError(
            f"Credential {credential_id} not found or not accessible."
        )
    try:
        data = decrypt_credential_data(credential.encrypted_data)
    except Exception as e:
        raise CredentialError(f"Failed to decrypt credential {credential_id}: {e}")
    required = ["aws_access_key_id", "aws_secret_access_key", "aws_region", "from_email"]
    missing = [k for k in required if not data.get(k)]
    if missing:
        raise CredentialError(
            f"Credential {credential_id} missing AWS SES fields: {missing}"
        )
    return data["from_email"]


def _render_template(template: str, variables: Dict[str, str]) -> str:
    """Replace {{ token }} placeholders — tolerates optional spaces around the name."""
    def replacer(m: re.Match) -> str:
        return variables.get(m.group(1), m.group(0))
    return re.sub(r'\{\{\s*(\w+)\s*\}\}', replacer, template)


def _build_template_list(db: Session, account_id: int) -> str:
    """Build a compact template catalogue for the LLM description."""
    try:
        templates = (
            db.query(EmailTemplate)
            .filter(EmailTemplate.account_id == account_id)
            .order_by(EmailTemplate.name)
            .all()
        )
        if not templates:
            return "No templates saved yet — create one in the Email Templates dashboard."
        lines = []
        for t in templates:
            vars_list = ", ".join(
                v["name"] for v in (t.variables or [])
            )
            var_str = f" | variables: {vars_list}" if vars_list else ""
            lines.append(f"  • ID {t.id}: {t.name}{var_str}")
        return "\n".join(lines)
    except Exception:
        return "(unable to list templates)"


async def create_send_template_email_with_ses_tool(
    tool_config: Dict[str, Any],
    account_id: int,
    db: Session,
    auth_token: Optional[str] = None,
    **kwargs,
) -> StructuredTool:
    """
    Create the HITL-gated send-template-email tool.

    Required tool_config keys:
        - credentialId: int — ID of the stored AWS SES credential
        - description:  str (optional) — extra LLM guidance
    """
    credential_id = tool_config.get("credentialId")
    if not credential_id:
        raise CredentialError(
            "Missing required field 'credentialId' in sendTemplateEmailWithSes tool config"
        )

    credential_account_id = kwargs.get("agent_owner_account_id", account_id)
    from_email = _verify_ses_credential(credential_id, credential_account_id, db)

    agent_id: Optional[int] = kwargs.get("agent_id")
    chat_session_id: Optional[int] = kwargs.get("chat_session_id_pk")

    # Build a live catalogue of available templates for the LLM
    template_catalogue = _build_template_list(db, credential_account_id)

    user_description = tool_config.get("description", "")
    description = (
        f"{user_description}\n\n" if user_description else ""
    ) + (
        "Send a beautifully designed HTML email using one of the saved templates.\n\n"
        "Available templates:\n"
        f"{template_catalogue}\n\n"
        "Instructions:\n"
        "- Pass the numeric template ID in the `template_id` field.\n"
        "- Pass all required variable values in `variables` as a JSON object "
        '  (e.g. {"first_name": "Alex", "body": "Here is your update…"}).\n'
        "- Leave any variable at its default by omitting it from `variables`.\n"
        "- The email requires human approval before it is sent."
    )

    from src.db.database import SessionLocal

    print(
        f"[SEND TEMPLATE EMAIL TOOL] HITL tool ready — "
        f"credential_id={credential_id}, account_id={credential_account_id}"
    )

    async def send_template_email_impl(
        to_email: str,
        template_id: int,
        variables: Dict[str, str],
    ) -> str:
        """Render the template and queue for human approval."""
        print(f"\n{'='*60}")
        print(f"[SEND TEMPLATE EMAIL TOOL] 📬 queuing template email")
        print(f"[SEND TEMPLATE EMAIL TOOL]   To          : {to_email}")
        print(f"[SEND TEMPLATE EMAIL TOOL]   Template ID : {template_id}")
        print(f"[SEND TEMPLATE EMAIL TOOL]   Variables   : {list(variables.keys())}")
        print(f"{'='*60}\n")

        # Fetch and render the template
        tool_db: Session = SessionLocal()
        try:
            tmpl = tool_db.query(EmailTemplate).filter(
                EmailTemplate.id == template_id,
                EmailTemplate.account_id == credential_account_id,
            ).first()
        finally:
            tool_db.close()

        if not tmpl:
            return json.dumps({
                "success": False,
                "error": f"Template ID {template_id} not found.",
            })

        # Merge defaults then apply caller-supplied values
        merged: Dict[str, str] = {}
        for v in (tmpl.variables or []):
            merged[v["name"]] = v.get("default", "")
        merged.update({k: str(val) for k, val in variables.items()})

        rendered_html = _render_template(tmpl.html_template, merged)
        rendered_subject = _render_template(tmpl.subject_template, merged)

        expires_at = datetime.now(timezone.utc) + timedelta(minutes=APPROVAL_TTL_MINUTES)

        approval_db: Session = SessionLocal()
        try:
            approval = PendingToolApproval(
                account_id=credential_account_id,
                agent_id=agent_id,
                chat_session_id=chat_session_id,
                tool_type="sendTemplateEmailWithSes",
                status="pending",
                payload={
                    "credential_id": credential_id,
                    "to_email": to_email,
                    "subject": rendered_subject,
                    "html_body": rendered_html,
                    "template_id": template_id,
                    "template_name": tmpl.name,
                    "variables": merged,
                },
                expires_at=expires_at,
            )
            approval_db.add(approval)
            approval_db.commit()
            approval_db.refresh(approval)
            approval_id = approval.id
            print(f"[SEND TEMPLATE EMAIL TOOL] ✅ PendingToolApproval created — id={approval_id}")
        except Exception as e:
            approval_db.rollback()
            import traceback; traceback.print_exc()
            return json.dumps({
                "success": False,
                "error": f"Failed to queue email for approval: {e}",
            })
        finally:
            approval_db.close()

        return json.dumps({
            HITL_SENTINEL_KEY: True,
            "approval_id": approval_id,
            "tool_type": "sendTemplateEmailWithSes",
            "preview": {
                "from_email": from_email,
                "to_email": to_email,
                "subject": rendered_subject,
                "html_body": rendered_html,
                "template_name": tmpl.name,
                "variables": merged,
            },
            "message": (
                f"Template email '{tmpl.name}' to {to_email} has been queued for "
                "human review. It will be sent only after the user approves it."
            ),
        })

    class SendTemplateEmailInput(BaseModel):
        to_email: str = Field(description="Recipient email address (e.g. user@example.com)")
        template_id: int = Field(description="Numeric ID of the email template to use")
        variables: Dict[str, str] = Field(
            default_factory=dict,
            description=(
                "Variable values to inject into the template. "
                "Keys must match the token names defined on the template. "
                'Example: {"first_name": "Alex", "body": "Your order has shipped."}'
            ),
        )

    return StructuredTool(
        func=send_template_email_impl,
        coroutine=send_template_email_impl,
        name="send_template_email_with_ses",
        description=description,
        args_schema=SendTemplateEmailInput,
    )
