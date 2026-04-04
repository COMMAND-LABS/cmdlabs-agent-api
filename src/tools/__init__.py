"""
Agent Tools

Registry + factory for creating LangChain tools from an agent config.
Add a new tool type by writing a builder module and registering it below.
"""
from .factory import create_tools_from_agent_config
from .registry import ToolRegistry
from .db_read import CredentialError

# ── Register all built-in tool types ────────────────────────────────────────
from .vector_search import create_vector_search_tool
from .vector_search_with_reranking import create_vector_search_with_reranking_tool
from .db_read import create_db_read_tool
from .db_write import create_db_write_tool
from .send_email_with_ses import create_send_email_with_ses_tool
from .send_html_email_with_ses import create_send_html_email_with_ses_tool
from .send_email_google_oauth import create_send_email_google_oauth_tool
from .send_email_google_smtp import create_send_email_google_smtp_tool

ToolRegistry.register("vectorSearch", create_vector_search_tool)
ToolRegistry.register("vectorSearchWithReranking", create_vector_search_with_reranking_tool)
ToolRegistry.register("dbTableRead", create_db_read_tool)
ToolRegistry.register("dbTableWrite", create_db_write_tool)
ToolRegistry.register("sendTxtEmailWithSes", create_send_email_with_ses_tool)
ToolRegistry.register("sendHtmlEmailWithSes", create_send_html_email_with_ses_tool)
ToolRegistry.register("sendTxtEmailWithGoogleOAuth", create_send_email_google_oauth_tool)
ToolRegistry.register("sendTxtEmailWithGoogleSmtp", create_send_email_google_smtp_tool)

__all__ = [
    "create_tools_from_agent_config",
    "ToolRegistry",
    "CredentialError",
]
