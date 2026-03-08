"""
TTS next-turn endpoint: one agent turn per request (approach 1).

Client sends prompt on first turn; after playing audio, sends stateToken to get
the next speaker. Returns JSON (no streaming) so the client can TTS and play,
then request the next turn.
"""

import base64
import json
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.exc import OperationalError

from src.deps import db_dependency, auth_dependency
from src.db.database import SessionLocal
from src.db.models import ChatSession, ChatMessage, Credential
from src.routers.credentials.encryption import get_credential_value
from src.routers.swarms.langgraph_schemas import (
    SwarmTtsNextTurnRequest,
    SwarmTtsNextTurnResponse,
)
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.multi_agent import route_message, stream_agent
from src.utils.langsmith import get_langsmith_callbacks

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

_DEFAULT_PROVIDER = "openai"
_langsmith_cbs = get_langsmith_callbacks("multi-agent-tts")


def _db_retry_once(db, label: str, fn):
    try:
        return fn()
    except OperationalError as e:
        text = str(e).lower()
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
        import time
        time.sleep(0.5)
        return fn()


def _credential_type_for(provider: str) -> Optional[str]:
    from src.db.service_name import ServiceName
    return {
        "openai": ServiceName.OPENAI_API_KEY,
        "anthropic": ServiceName.ANTHROPIC_API_KEY,
        "ollama": None,
    }.get(provider)


def _create_llm(provider: str, model: str, api_key: str, *, streaming: bool):
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, streaming=streaming, temperature=0.7, callbacks=_langsmith_cbs)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=api_key, streaming=streaming, temperature=0.7, callbacks=_langsmith_cbs)
    raise ValueError(f"Unsupported provider: {provider}")


def _store_user_msg(chat_session_id: int, prompt: str) -> None:
    from src.routers.agents.helpers.message_history import store_user_message
    db = SessionLocal()
    try:
        store_user_message(db, chat_session_id, prompt)
    finally:
        db.close()


def _store_ai_msg(chat_session_id: int, content: str, agent_name: Optional[str] = None) -> None:
    from src.routers.agents.helpers.message_history import store_ai_message
    db = SessionLocal()
    try:
        store_ai_message(db, chat_session_id, content, agent_name=agent_name)
    finally:
        db.close()


def _encode_state(session_id: str, pending_speakers: List[str]) -> str:
    payload = {"sessionId": session_id, "pendingSpeakers": pending_speakers}
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def _decode_state(token: str) -> Dict[str, Any]:
    try:
        raw = base64.urlsafe_b64decode(token.encode())
        return json.loads(raw.decode())
    except Exception:
        return {}


@router.post("/tts/next-turn", response_model=SwarmTtsNextTurnResponse)
@limiter.limit("200/minute")
async def swarm_tts_next_turn(
    request: Request,
    body: SwarmTtsNextTurnRequest,
    db: db_dependency,
    auth: auth_dependency,
) -> SwarmTtsNextTurnResponse:
    """Return one agent's turn. Send prompt for first turn; send stateToken after playback for next."""
    account_id = int(auth["id"]) if isinstance(auth["id"], str) else auth["id"]

    # Resolve API key
    provider = _DEFAULT_PROVIDER
    required_cred = _credential_type_for(provider)
    if required_cred:
        cred = _db_retry_once(
            db,
            "load credential",
            lambda: db.query(Credential).filter(
                Credential.account_id == account_id,
                Credential.service_name == required_cred,
            ).first(),
        )
        if not cred:
            raise HTTPException(status_code=400, detail="API key required. Add your OpenAI API key in account settings.")
        api_key = get_credential_value(cred, "api_key")
    else:
        api_key = ""

    sw = body.swarm
    all_display_names = [w.agentName for w in sw.workers]

    agent_configs: Dict[str, Dict] = {}
    agent_list: List[Dict[str, str]] = []
    for w in sw.workers:
        base_prompt = w.systemPrompt or f"You are {w.agentName}."
        participants = ", ".join(
            f"{n} (you)" if n == w.agentName else n
            for n in all_display_names
        )
        full_prompt = (
            f"{base_prompt}\n\n"
            f"Group conversation participants: Human, {participants}.\n"
            "- Speak naturally as yourself. Only say your own words.\n"
            "- NEVER speak on behalf of other participants — let them answer for themselves.\n"
            "- Do NOT end every message with a question. Sometimes just share a thought, react, or make a statement.\n"
            "- Be direct. Keep it concise. A few sentences is usually enough."
        )
        agent_configs[w.agentName] = {
            "system_prompt": full_prompt,
            "llm": _create_llm(provider, w.modelName, api_key, streaming=True),
        }
        agent_list.append({"name": w.agentName, "description": w.agentDescription or ""})

    router_llm = _create_llm(provider, sw.supervisor.modelName, api_key, streaming=False)

    try:
        session_uuid = uuid.UUID(body.sessionId)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid sessionId format.")

    session = _db_retry_once(
        db,
        "load session",
        lambda: db.query(ChatSession).filter(
            ChatSession.session_id == session_uuid,
            ChatSession.account_id == account_id,
        ).first(),
    )
    if not session:
        session = ChatSession(
            session_id=session_uuid,
            agent_id=None,
            account_id=account_id,
            title="Multi-agent TTS chat",
        )
        db.add(session)
        db.commit()
        db.refresh(session)

    db_messages = _db_retry_once(
        db,
        "load messages",
        lambda: db.query(ChatMessage).filter(ChatMessage.chat_session_id == session.id)
        .order_by(ChatMessage.created_at.asc())
        .all(),
    )

    history: List[Dict] = []
    for msg in db_messages:
        md = msg.message
        if isinstance(md, dict) and "role" in md and "content" in md:
            role = "user" if md["role"] == "human" else "assistant"
            entry: Dict = {"role": role, "content": md["content"]}
            if role == "assistant" and md.get("agentName"):
                entry["agent_name"] = md["agentName"]
            history.append(entry)

    chat_session_id = session.id
    db.close()

    # Continuation: stateToken present
    if body.stateToken:
        state = _decode_state(body.stateToken)
        pending_speakers = state.get("pendingSpeakers") or []
        if state.get("sessionId") != body.sessionId:
            raise HTTPException(status_code=400, detail="stateToken does not match sessionId.")
        if not pending_speakers:
            return SwarmTtsNextTurnResponse(agentName="", content="", stateToken=None, done=True)

        name = pending_speakers[0]
        cfg = agent_configs.get(name)
        if not cfg:
            return SwarmTtsNextTurnResponse(agentName="", content="", stateToken=None, done=True)

        full_text = ""
        async for token in stream_agent(
            system_prompt=cfg["system_prompt"],
            history=history,
            prompt="Continue the conversation.",
            llm=cfg["llm"],
            agent_name=name,
        ):
            full_text += token

        try:
            _store_ai_msg(chat_session_id, full_text, agent_name=name)
        except Exception:
            pass

        new_pending = pending_speakers[1:]
        new_token = _encode_state(body.sessionId, new_pending) if new_pending else None
        return SwarmTtsNextTurnResponse(
            agentName=name,
            content=full_text,
            stateToken=new_token,
            done=len(new_pending) == 0,
        )

    # First turn: prompt required
    if not (body.prompt or "").strip():
        raise HTTPException(status_code=400, detail="prompt is required when stateToken is not provided.")

    prompt = (body.prompt or "").strip()
    try:
        _store_user_msg(chat_session_id, prompt)
    except Exception:
        pass

    speakers = await route_message(prompt, history, agent_list, router_llm)
    if not speakers:
        return SwarmTtsNextTurnResponse(agentName="", content="", stateToken=None, done=True)

    name = speakers[0]
    cfg = agent_configs.get(name)
    if not cfg:
        return SwarmTtsNextTurnResponse(agentName="", content="", stateToken=None, done=True)

    full_text = ""
    async for token in stream_agent(
        system_prompt=cfg["system_prompt"],
        history=history,
        prompt=prompt,
        llm=cfg["llm"],
        agent_name=name,
    ):
        full_text += token

    try:
        _store_ai_msg(chat_session_id, full_text, agent_name=name)
    except Exception:
        pass

    new_pending = speakers[1:]
    new_token = _encode_state(body.sessionId, new_pending) if new_pending else None
    return SwarmTtsNextTurnResponse(
        agentName=name,
        content=full_text,
        stateToken=new_token,
        done=len(new_pending) == 0,
    )
