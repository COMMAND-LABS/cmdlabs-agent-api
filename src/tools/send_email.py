"""
Send Plain-Text Email Tool (AWS SES)

Allows agents to send plain-text emails through an account-owned AWS SES
credential.  The credential must contain the following decrypted fields:
    - aws_access_key_id
    - aws_secret_access_key
    - aws_region          (e.g. "us-east-1")
    - from_email          (a verified SES sender identity)
"""
from typing import Dict, Any, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.db.models import Credential
from src.routers.credentials.encryption import decrypt_credential_data


class CredentialError(Exception):
    """Raised when there is an issue with the AWS SES credential."""
    pass


def get_ses_config(credential_id: int, account_id: int, db: Session) -> Dict[str, str]:
    """
    Retrieve and decrypt the AWS SES configuration from a stored credential.

    Returns a dict with keys: aws_access_key_id, aws_secret_access_key,
    aws_region, from_email.

    Raises:
        CredentialError: If credential is missing, wrong type, or missing keys.
    """
    credential = db.query(Credential).filter(
        Credential.id == credential_id,
        Credential.account_id == account_id,
    ).first()

    if not credential:
        raise CredentialError(
            f"Credential with ID {credential_id} not found. "
            "It may have been deleted or you do not have access to it."
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

    return {k: data[k] for k in required}


async def create_send_email_tool(
    tool_config: Dict[str, Any],
    account_id: int,
    db: Session,
    auth_token: Optional[str] = None,
    **kwargs,
) -> StructuredTool:
    """
    Create a send-plain-text-email tool backed by AWS SES.

    Args:
        tool_config: Tool configuration including:
            - credentialId: ID of stored AWS SES credential
            - name:         Optional custom tool name (default: send_txt_email)
            - description:  Guidance for the LLM on when/how to use this tool
        account_id: Account ID for credential ownership check
        db:          Database session
        auth_token:  Unused (present for interface parity)
        **kwargs:    agent_owner_account_id forwarded for shared-agent support

    Returns:
        StructuredTool that sends plain-text emails via AWS SES.

    Example tool_config:
        {
            "type": "sendTxtEmail",
            "credentialId": 12,
            "name": "send_followup_email",
            "description": "Send a plain-text follow-up email to a prospect."
        }
    """
    credential_id = tool_config.get("credentialId")
    tool_name = (tool_config.get("name") or "send_txt_email").strip()
    description = (
        tool_config.get("description")
        or "Send a plain-text email to a recipient using AWS SES."
    )

    if not credential_id:
        raise CredentialError(
            "Missing required field 'credentialId' in sendTxtEmail tool configuration"
        )

    # Use the agent owner's credentials so the tool works for shared agents
    # (same policy as dbTableRead — the owner's SES identity is used).
    credential_account_id = kwargs.get("agent_owner_account_id", account_id)

    ses_config = get_ses_config(credential_id, credential_account_id, db)

    print(
        f"[SEND EMAIL TOOL] Tool '{tool_name}' ready "
        f"(from: {ses_config['from_email']}, region: {ses_config['aws_region']})"
    )

    # Define the send implementation — created once at tool-build time so the
    # SES client is not recreated on every invocation.
    async def send_email_impl(to_email: str, subject: str, body: str) -> Dict[str, Any]:
        """Send a plain-text email via AWS SES."""
        print(f"\n{'='*60}")
        print(f"[SEND EMAIL TOOL] 🚀 TOOL INVOKED: {tool_name}")
        print(f"[SEND EMAIL TOOL] 📧 To: {to_email}")
        print(f"[SEND EMAIL TOOL] 📋 Subject: {subject}")
        print(f"[SEND EMAIL TOOL] 📝 Body length: {len(body)} chars")
        print(f"{'='*60}\n")

        try:
            import boto3

            ses_client = boto3.client(
                "ses",
                region_name=ses_config["aws_region"],
                aws_access_key_id=ses_config["aws_access_key_id"],
                aws_secret_access_key=ses_config["aws_secret_access_key"],
            )

            response = ses_client.send_email(
                Source=ses_config["from_email"],
                Destination={"ToAddresses": [to_email]},
                Message={
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
                },
            )

            message_id = response.get("MessageId", "unknown")
            print(f"[SEND EMAIL TOOL] ✅ Email sent — MessageId: {message_id}")
            print(f"{'='*60}\n")

            return {
                "success": True,
                "message_id": message_id,
                "to": to_email,
                "subject": subject,
            }

        except Exception as e:
            print(f"\n[SEND EMAIL TOOL] ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return {"success": False, "error": str(e)}

    class SendEmailInput(BaseModel):
        to_email: str = Field(
            description="Recipient email address (e.g. user@example.com)"
        )
        subject: str = Field(
            description="Email subject line"
        )
        body: str = Field(
            description="Plain-text email body"
        )

    return StructuredTool(
        func=send_email_impl,
        coroutine=send_email_impl,
        name=tool_name,
        description=description,
        args_schema=SendEmailInput,
    )
