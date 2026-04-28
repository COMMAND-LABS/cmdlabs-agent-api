"""Shared agent context preparation.

Extracts the common setup logic used by both the streaming and
non-streaming completion endpoints into a single place.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import Request
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent, create_tool_calling_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool

from src.db.database import SessionLocal
from src.db.models import Agent, ChatSession, ChatMessage, Credential
from src.db.retry import db_retry_once
from src.routers.agents.access import load_agent_with_access_check
from src.routers.agents.helpers import (
    build_message_history,
    store_user_message,
    store_ai_message,
    extract_auth_token,
    get_model_config,
    create_llm,
    get_required_credential_type,
)
from src.routers.credentials.encryption import get_credential_value
from src.tools import create_tools_from_agent_config, CredentialError
from src.utils.pdf_to_images import build_pdf_message
from src.utils.template_variables import resolve_template_variables, build_variable_context


@dataclass
class AgentContext:
    """Everything the streaming / non-streaming endpoints need after setup."""

    agent: Agent
    account_id: int
    provider: str
    model_name: str
    llm: BaseChatModel
    tools: List[StructuredTool]
    prompt_template: ChatPromptTemplate
    memory: ConversationBufferMemory
    message_history: ChatMessageHistory
    agent_executor: Optional[AgentExecutor]
    agent_input: Any
    chat_session_id: int
    session_uuid: uuid.UUID
    user_email: str
    prompt: str
    pdf_filename: Optional[str]
    callbacks: list


class AgentSetupError(Exception):
    """Raised when agent setup fails with a user-facing message."""

    def __init__(self, title: str, detail: str):
        self.title = title
        self.detail = detail
        super().__init__(f"{title}: {detail}")


async def prepare_agent_context(
    *,
    agent_id: int,
    session_id: str,
    prompt: str,
    db,
    auth: dict,
    request: Request,
    callbacks: list,
    streaming: bool = True,
    pdf_base64: Optional[str] = None,
    pdf_filename: Optional[str] = None,
    pdf_use_vision: bool = False,
) -> AgentContext:
    """Build the full agent context shared by stream and completion endpoints.

    Raises ``AgentSetupError`` for any user-facing failure.
    """
    account_id = auth["id"]

    agent = db_retry_once(
        db, "load agent",
        lambda: load_agent_with_access_check(db, account_id, agent_id),
    )
    if not agent:
        raise AgentSetupError("Agent not found", "The specified agent was not found or you do not have access.")
    if not agent.config:
        raise AgentSetupError("Invalid agent configuration", "Agent configuration is missing.")

    config_data = agent.config.get("data", {})
    system_prompt_raw = config_data.get("systemPrompt", "You are a helpful assistant.")
    var_context = build_variable_context(agent_name=agent.name)
    system_prompt = resolve_template_variables(system_prompt_raw, var_context).replace("{", "{{").replace("}", "}}")

    model_config = get_model_config(agent.config)
    provider = model_config["provider"]
    model_name = model_config["model"]

    # --- Credentials ---
    required_credential_type = get_required_credential_type(provider)
    credentials: Dict[str, str] = {}
    if required_credential_type:
        credential = db_retry_once(
            db, "load provider credential",
            lambda: db.query(Credential).filter(
                Credential.account_id == account_id,
                Credential.credential_type == required_credential_type,
            ).first(),
        )
        if not credential:
            raise AgentSetupError(
                f"{provider.title()} API key required",
                f"Please add your {provider.title()} API key in account settings to use {model_name}.",
            )
        try:
            credentials[provider] = get_credential_value(credential, "api_key")
        except Exception as exc:
            raise AgentSetupError("Failed to retrieve API key", str(exc))

    # --- LLM ---
    try:
        llm, _ = create_llm(
            model_config=model_config,
            credentials=credentials,
            streaming=streaming,
            temperature=0,
        )
    except ValueError as exc:
        raise AgentSetupError("LLM initialization failed", str(exc))

    # --- Session ---
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise AgentSetupError("Invalid sessionId format", "The sessionId must be a valid UUID format.")

    session = db_retry_once(
        db, "load chat session",
        lambda: db.query(ChatSession).filter(
            ChatSession.session_id == session_uuid,
            ChatSession.account_id == account_id,
        ).first(),
    )
    if not session:
        try:
            session = ChatSession(
                session_id=session_uuid,
                agent_id=agent_id,
                account_id=account_id,
                title=f"Chat with Agent {agent_id}",
            )
            db.add(session)
            db.commit()
            db.refresh(session)
        except Exception as exc:
            db.rollback()
            raise AgentSetupError("Failed to create session", f"Could not create chat session: {exc}")

    # --- History ---
    db_messages = db_retry_once(
        db, "load chat messages",
        lambda: db.query(ChatMessage).filter(
            ChatMessage.chat_session_id == session.id,
        ).order_by(ChatMessage.created_at.asc()).all(),
    )
    message_history = build_message_history(db_messages)
    auth_token = extract_auth_token(request, auth)

    # --- Tools ---
    try:
        tools = await create_tools_from_agent_config(
            agent_config=agent.config,
            account_id=account_id,
            db=db,
            auth_token=auth_token,
            request=request,
            chat_session_id=session_uuid,
            agent_id=agent_id,
            chat_session_id_pk=session.id,
            agent_owner_account_id=agent.account_id,
        )
    except CredentialError as exc:
        raise AgentSetupError("Tool configuration error", str(exc))
    except ValueError as exc:
        raise AgentSetupError("Invalid tool configuration", str(exc))

    # --- Prompt template + agent ---
    if tools:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        tagged_llm = llm.with_config({"tags": ["agent_llm"]}) if streaming else llm
        if provider == "openai":
            agent_langchain = create_openai_tools_agent(tagged_llm, tools, prompt_template)
        else:
            agent_langchain = create_tool_calling_agent(tagged_llm, tools, prompt_template)
    else:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        agent_langchain = None

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        return_messages=True,
        output_key="output" if tools else None,
    )

    user_email = auth.get("email", "unknown")
    agent_executor: Optional[AgentExecutor] = None
    if tools and agent_langchain:
        agent_executor = AgentExecutor(
            agent=agent_langchain,
            tools=tools,
            memory=memory,
            max_iterations=25 if streaming else 10,
        ).with_config({
            "run_name": "Agent",
            "callbacks": callbacks,
            "metadata": {"user_email": user_email, "agent_id": agent_id, "session_id": str(session_uuid)},
            "tags": [f"user:{user_email}", f"agent:{agent_id}"],
        })

    # --- Agent input (text or PDF) ---
    if pdf_base64:
        agent_input = build_pdf_message(
            prompt=prompt,
            pdf_base64=pdf_base64,
            pdf_filename=pdf_filename,
            use_vision=pdf_use_vision,
            max_pages=10 if pdf_use_vision else 50,
        )
    else:
        agent_input = prompt

    # Release the DB connection before the long-running LLM call
    chat_session_id = session.id
    db.close()

    return AgentContext(
        agent=agent,
        account_id=account_id,
        provider=provider,
        model_name=model_name,
        llm=llm,
        tools=tools,
        prompt_template=prompt_template,
        memory=memory,
        message_history=message_history,
        agent_executor=agent_executor,
        agent_input=agent_input,
        chat_session_id=chat_session_id,
        session_uuid=session_uuid,
        user_email=user_email,
        prompt=prompt,
        pdf_filename=pdf_filename,
        callbacks=callbacks,
    )


# ---------------------------------------------------------------------------
# Short-lived DB session wrappers for message persistence
# ---------------------------------------------------------------------------

def persist_user_message(chat_session_id: int, prompt: str, pdf_filename: Optional[str] = None):
    """Write user message using a short-lived DB session."""
    db = SessionLocal()
    try:
        store_user_message(db, chat_session_id, prompt, pdf_filename)
    finally:
        db.close()


def persist_ai_message(chat_session_id: int, content: str, tool_calls=None):
    """Write AI message using a short-lived DB session."""
    db = SessionLocal()
    try:
        store_ai_message(db, chat_session_id, content, tool_calls)
    finally:
        db.close()
