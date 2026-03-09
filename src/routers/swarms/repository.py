from __future__ import annotations

import time
import uuid
from typing import Optional

from fastapi import HTTPException
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from src.db.models import ChatMessage, ChatSession, Credential
from src.routers.agents.helpers.message_history import store_ai_message, store_user_message
from src.routers.credentials.encryption import get_credential_value
from src.routers.swarms.langgraph_schemas import LanggraphSwarmConfigInput
from src.routers.swarms.types import ConversationEntry


DEFAULT_PROVIDER = "openai"


def db_retry_once(db: Session, label: str, fn):
    try:
        return fn()
    except OperationalError as exc:
        text = str(exc).lower()
        if not (
            "ssl connection has been closed unexpectedly" in text
            or "server closed the connection unexpectedly" in text
            or "connection reset by peer" in text
        ):
            raise
        try:
            db.rollback()
        except Exception:
            pass
        db.close()
        time.sleep(0.5)
        return fn()


def credential_type_for(provider: str) -> Optional[str]:
    from src.db.service_name import ServiceName

    return {
        "openai": ServiceName.OPENAI_API_KEY,
        "anthropic": ServiceName.ANTHROPIC_API_KEY,
        "ollama": None,
    }.get(provider)


def resolve_api_key(db: Session, *, account_id: int, provider: str) -> str:
    required_cred = credential_type_for(provider)
    if not required_cred:
        return ""

    credential = db_retry_once(
        db,
        "load credential",
        lambda: db.query(Credential).filter(
            Credential.account_id == account_id,
            Credential.service_name == required_cred,
        ).first(),
    )
    if not credential:
        raise HTTPException(
            status_code=400,
            detail=f"API key required. Add your {provider.title()} API key in account settings.",
        )
    return get_credential_value(credential, "api_key")


def load_or_create_chat_session(
    db: Session,
    *,
    session_id: str,
    account_id: int,
    title: str,
) -> ChatSession:
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid sessionId format.") from exc

    session = db_retry_once(
        db,
        "load session",
        lambda: db.query(ChatSession).filter(
            ChatSession.session_id == session_uuid,
            ChatSession.account_id == account_id,
        ).first(),
    )
    if session:
        return session

    session = ChatSession(
        session_id=session_uuid,
        agent_id=None,
        account_id=account_id,
        title=title,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def load_session_messages(db: Session, *, chat_session_id: int) -> list[ChatMessage]:
    return db_retry_once(
        db,
        "load messages",
        lambda: db.query(ChatMessage)
        .filter(ChatMessage.chat_session_id == chat_session_id)
        .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
        .all(),
    )


def build_history(db_messages: list[ChatMessage]) -> list[ConversationEntry]:
    history: list[ConversationEntry] = []
    for message in db_messages:
        payload = message.message
        if isinstance(payload, dict) and "role" in payload and "content" in payload:
            role = "user" if payload["role"] == "human" else "assistant"
            history.append(
                ConversationEntry(
                    role=role,
                    content=payload["content"],
                    agent_name=payload.get("agentName") if role == "assistant" else None,
                )
            )
    return history


def persist_user_message(db: Session, *, chat_session_id: int, prompt: str) -> ChatMessage:
    message = store_user_message(db, chat_session_id, prompt)
    if not message:
        raise RuntimeError("Failed to persist user message.")
    return message


def persist_ai_message(
    db: Session,
    *,
    chat_session_id: int,
    content: str,
    agent_name: Optional[str] = None,
) -> ChatMessage:
    message = store_ai_message(db, chat_session_id, content, agent_name=agent_name)
    if not message:
        raise RuntimeError("Failed to persist AI message.")
    return message


def latest_message_id(db_messages: list[ChatMessage]) -> Optional[int]:
    if not db_messages:
        return None
    return db_messages[-1].id


def count_assistant_messages_since_last_user(history: list[ConversationEntry]) -> int:
    count = 0
    for entry in reversed(history):
        if entry.role == "assistant":
            count += 1
            continue
        if entry.role == "user":
            break
    return count


def build_account_id(auth: dict) -> int:
    account_id = auth["id"]
    return int(account_id) if isinstance(account_id, str) else account_id


def ensure_matching_swarm(_: LanggraphSwarmConfigInput) -> None:
    """Placeholder for future DB-backed swarm validation."""
    return None
