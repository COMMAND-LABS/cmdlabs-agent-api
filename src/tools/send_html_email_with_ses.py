"""
Send Templated HTML Email Tool via AWS SES — Human-in-the-Loop (HITL) variant.

Preferred mode — template-based:
    The agent picks a saved EmailTemplate by ID and supplies variable values.
    The template is rendered server-side ({{token}} substitution), producing
    a vetted, production-grade HTML email.  This is strongly preferred over
    supplying raw HTML because it avoids layout regressions and keeps brand
    consistency.

Fallback mode — raw HTML:
    If `template_id` is not provided the agent may supply a complete HTML
    document in `html_body`.  This is an escape hatch for one-off emails that
    genuinely have no matching template.

HITL flow (both modes):
  1. Writes a PendingToolApproval row with the rendered html_body in payload.
  2. Returns a HITL sentinel so the streaming layer emits a
     ``tool_approval_required`` SSE event.  The chat card shows a live iframe
     preview of the rendered email.
  3. Returns a human-readable "pending" message to the LLM.
"""
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from src.db.models import Credential, EmailTemplate, PendingToolApproval
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
            f"Credential {credential_id} missing required AWS SES fields: {missing}"
        )
    return data["from_email"]


def _render(template: str, variables: Dict[str, str]) -> str:
    """Replace {{ token }} placeholders — tolerates optional spaces around the name."""
    return re.sub(
        r'\{\{\s*(\w+)\s*\}\}',
        lambda m: variables.get(m.group(1), m.group(0)),
        template,
    )


def _build_template_catalogue(db: Session, account_id: int) -> str:
    try:
        templates = (
            db.query(EmailTemplate)
            .filter(EmailTemplate.account_id == account_id)
            .order_by(EmailTemplate.name)
            .all()
        )
        if not templates:
            return "  (no templates saved yet — create one in the Email Templates dashboard)"
        lines = []
        for t in templates:
            vars_str = ", ".join(v["name"] for v in (t.variables or []))
            lines.append(
                f"  • ID {t.id}: {t.name}"
                + (f"  |  variables: {vars_str}" if vars_str else "")
                + (f"\n    {t.description}" if t.description else "")
            )
        return "\n".join(lines)
    except Exception:
        return "  (unable to list templates)"


async def create_send_html_email_with_ses_tool(
    tool_config: Dict[str, Any],
    account_id: int,
    db: Session,
    auth_token: Optional[str] = None,
    **kwargs,
) -> StructuredTool:
    """
    Create the HITL-gated HTML email tool.

    Required tool_config keys:
        - credentialId: int — ID of the stored AWS SES credential
        - description:  str (optional) — extra LLM guidance
    """
    credential_id = tool_config.get("credentialId")
    if not credential_id:
        raise CredentialError(
            "Missing required field 'credentialId' in sendHtmlEmailWithSes tool configuration"
        )

    credential_account_id = kwargs.get("agent_owner_account_id", account_id)
    from_email = _verify_ses_credential(credential_id, credential_account_id, db)

    agent_id: Optional[int] = kwargs.get("agent_id")
    chat_session_id: Optional[int] = kwargs.get("chat_session_id_pk")

    # Build a live catalogue of templates for the LLM description
    catalogue = _build_template_catalogue(db, credential_account_id)

    user_description = tool_config.get("description", "")
    description = (
        (f"{user_description}\n\n" if user_description else "")
        + "Send a professional HTML email via AWS SES.  The email requires human "
        "approval before it is delivered.\n\n"
        "━━ PREFERRED: use a saved template ━━\n"
        "Pass `template_id` (integer) and `variables` (object with token→value pairs).\n"
        "The template is rendered server-side — consistent layout, tracked opens.\n\n"
        f"Available templates:\n{catalogue}\n\n"
        "━━ FALLBACK: raw HTML ━━\n"
        "Only use `html_body` when no suitable template exists.  Omit `template_id`.\n"
        "The HTML must be a self-contained, inline-CSS, table-layout document ≤600 px wide."
    )

    from src.db.database import SessionLocal

    print(
        f"[SEND HTML EMAIL TOOL] ready — "
        f"credential_id={credential_id}, account_id={credential_account_id}"
    )

    async def queued_send(
        to_email: str,
        template_id: Optional[int] = None,
        variables: Optional[Dict[str, str]] = None,
        html_body: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> str:
        """Resolve template / raw HTML, then create the PendingToolApproval."""
        template_name: Optional[str] = None
        merged_variables: Dict[str, str] = {}

        # ── Template mode ──────────────────────────────────────────────────────
        if template_id is not None:
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
                    "error": (
                        f"Template ID {template_id} not found. "
                        "Use one of the IDs shown in this tool's description."
                    ),
                })

            for v in (tmpl.variables or []):
                merged_variables[v["name"]] = v.get("default", "")
            merged_variables.update({k: str(val) for k, val in (variables or {}).items()})

            html_body = _render(tmpl.html_template, merged_variables)
            # Subject always comes from the template — LLM must not override it
            subject = _render(tmpl.subject_template, merged_variables)
            template_name = tmpl.name

        # ── Raw HTML mode ──────────────────────────────────────────────────────
        else:
            if not html_body or not html_body.strip():
                return json.dumps({
                    "success": False,
                    "error": (
                        "Either 'template_id' or 'html_body' must be provided. "
                        "Prefer template_id — see the available templates in this tool's description."
                    ),
                })
            if not subject or not subject.strip():
                return json.dumps({
                    "success": False,
                    "error": "'subject' is required when not using a template.",
                })

        print(f"\n{'='*60}")
        print(f"[SEND HTML EMAIL TOOL] 📬 queuing for approval")
        print(f"[SEND HTML EMAIL TOOL]   To          : {to_email}")
        print(f"[SEND HTML EMAIL TOOL]   Subject     : {subject}")
        print(f"[SEND HTML EMAIL TOOL]   template_id : {template_id}")
        print(f"[SEND HTML EMAIL TOOL]   HTML bytes  : {len(html_body or '')}")
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
                    # Template metadata stored for reference / audit
                    "template_id": template_id,
                    "template_name": template_name,
                    "variables": merged_variables if merged_variables else None,
                },
                expires_at=expires_at,
            )
            approval_db.add(approval)
            approval_db.commit()
            approval_db.refresh(approval)
            approval_id = approval.id
            print(f"[SEND HTML EMAIL TOOL] ✅ PendingToolApproval id={approval_id}")
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
            "tool_type": "sendHtmlEmailWithSes",
            "preview": {
                "from_email": from_email,
                "to_email": to_email,
                "subject": subject,
                "html_body": html_body,
                "template_name": template_name,
                "variables": merged_variables if merged_variables else None,
            },
            "message": (
                f"{'Template ' + repr(template_name) + ' email' if template_name else 'HTML email'} "
                f"to {to_email} has been queued for human review. "
                "It will be sent only after the user approves it."
            ),
        })

    class SendHtmlEmailInput(BaseModel):
        to_email: str = Field(
            description="Recipient email address (e.g. user@example.com)"
        )
        template_id: Optional[int] = Field(
            default=None,
            description=(
                "ID of a saved email template — STRONGLY PREFERRED. "
                "The subject line and HTML body are derived entirely from the template; "
                "do NOT provide a separate subject when using a template. "
                "Look up the available templates listed in this tool's description."
            ),
        )
        variables: Optional[Dict[str, str]] = Field(
            default=None,
            description=(
                "Variable values to inject into the template. "
                "Keys must match the token names defined on the template. "
                'Example: {"first_name": "Alex", "body": "Your order shipped today."}'
            ),
        )
        html_body: Optional[str] = Field(
            default=None,
            description=(
                "Complete, self-contained HTML email body. "
                "Only use when no suitable template exists. "
                "Must use inline CSS and a table-based layout ≤ 600 px wide."
            ),
        )
        subject: Optional[str] = Field(
            default=None,
            description=(
                "Email subject line — required ONLY when using raw html_body (no template). "
                "When template_id is provided the subject is set by the template automatically; "
                "do not provide this field."
            ),
        )

        @model_validator(mode="after")
        def validate_inputs(self) -> "SendHtmlEmailInput":
            if self.template_id is None:
                if not (self.html_body and self.html_body.strip()):
                    raise ValueError(
                        "Provide either 'template_id' (preferred) or 'html_body' (fallback)."
                    )
                if not (self.subject and self.subject.strip()):
                    raise ValueError(
                        "'subject' is required when using raw html_body instead of a template."
                    )
            return self

    return StructuredTool(
        func=queued_send,
        coroutine=queued_send,
        name="send_html_email_with_ses",
        description=description,
        args_schema=SendHtmlEmailInput,
    )
