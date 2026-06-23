"""Send Plain-Text Email Tool via Google Gmail API (OAuth) — HITL variant."""

from typing import Any

from langchain_core.tools import StructuredTool
from sqlalchemy.orm import Session

from src.tools.hitl_email_base import create_hitl_plain_email_tool


async def create_send_email_google_oauth_tool(
    tool_config: dict[str, Any],
    account_id: int,
    db: Session,
    auth_token: str | None = None,
    **kwargs,
) -> StructuredTool:
    return await create_hitl_plain_email_tool(
        tool_config=tool_config,
        account_id=account_id,
        db=db,
        tool_type="sendTxtEmailWithGoogleOAuth",
        tool_name="send_txt_email_with_google_oauth",
        required_credential_fields=["client_id", "client_secret", "refresh_token", "from_email"],
        provider_label="Google OAuth",
        default_description=(
            "Send a plain-text email to a recipient using Google Gmail (OAuth). "
            "The email will be reviewed by a human before it is delivered."
        ),
        **kwargs,
    )
