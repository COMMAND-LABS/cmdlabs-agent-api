"""
TTS next-turn endpoint: one agent turn per request.

Client sends prompt on first turn; after playing audio, sends stateToken to get
the next speaker. Returns JSON (no streaming) so the client can TTS and play,
then request the next turn.

Flow per user message (max 5 agent responses):
  1. User sends prompt → router picks first speaker → agent responds.
  2. Client sends stateToken → router picks next speaker → agent responds.
  3. Repeat until router returns [] or 5 responses reached.

All parties see the same shared history — no synthetic prompts are injected.
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

from src.multi_agent.session_logger import SessionLogger
from src.routers.swarms.plain_llm import generate_agent_reply_plain, route_next_speaker_plain

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

_DEFAULT_PROVIDER = "openai"
_MAX_RESPONSES_PER_TURN = 5


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


def _encode_state(session_id: str, response_count: int) -> str:
    return base64.urlsafe_b64encode(
        json.dumps({"sessionId": session_id, "responseCount": response_count}).encode()
    ).decode()


def _decode_state(token: str) -> Dict[str, Any]:
    try:
        return json.loads(base64.urlsafe_b64decode(token.encode()).decode())
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
    """Return one agent's turn, then route to decide if someone else should speak."""
    account_id = int(auth["id"]) if isinstance(auth["id"], str) else auth["id"]

    # --- resolve API key ---
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

    # --- build agent configs ---
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
            "- Speak naturally as yourself, in your own voice.\n"
            "- Only say your own words. Do not speak for other participants.\n"
            "- If the latest message is a simple greeting or direct check-in, a short natural reply is enough.\n"
            "- Do not force a bigger discussion unless the conversation genuinely calls for it.\n"
            "- Be concise. A few sentences is usually enough."
        )
        agent_configs[w.agentName] = {
            "system_prompt": full_prompt,
            "model": w.modelName,
        }
        agent_list.append({"name": w.agentName, "description": w.agentDescription or ""})

    # --- session & history ---
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
    slog = SessionLogger(body.sessionId)
    db.close()

    # --- determine response_count ---
    if body.stateToken:
        state = _decode_state(body.stateToken)
        if state.get("sessionId") != body.sessionId:
            raise HTTPException(status_code=400, detail="stateToken does not match sessionId.")
        response_count = state.get("responseCount", 0)
    else:
        response_count = 0

    # --- first turn: persist user message and add to history ---
    if not body.stateToken:
        if not (body.prompt or "").strip():
            raise HTTPException(status_code=400, detail="prompt is required when stateToken is not provided.")
        prompt = (body.prompt or "").strip()
        try:
            _store_user_msg(chat_session_id, prompt)
        except Exception:
            pass
        history.append({"role": "user", "content": prompt})

    # --- route: who should speak next? ---
    speakers, route_reason = await route_next_speaker_plain(
        history=history,
        agents=agent_list,
        provider=provider,
        model=sw.supervisor.modelName,
        api_key=api_key,
        session_logger=slog,
    )
    slog.log_route(history[-1].get("content", "")[:80], speakers, route_reason)

    if not speakers:
        return SwarmTtsNextTurnResponse(agentName="", content="", stateToken=None, done=True)

    name = speakers[0]
    cfg = agent_configs.get(name)
    if not cfg:
        return SwarmTtsNextTurnResponse(agentName="", content="", stateToken=None, done=True)

    # --- generate agent response (agent sees the full shared history) ---
    slog.log_agent_start(name, len(history), history[-1].get("content", "")[:80])
    t0 = SessionLogger.timer()
    full_text = await generate_agent_reply_plain(
        provider=provider,
        model=cfg["model"],
        api_key=api_key,
        system_prompt=cfg["system_prompt"],
        history=history,
        agent_name=name,
        session_logger=slog,
    )

    slog.log_agent_end(name, full_text, SessionLogger.timer() - t0)

    try:
        _store_ai_msg(chat_session_id, full_text, agent_name=name)
    except Exception:
        pass

    response_count += 1

    # --- decide if we're done ---
    if response_count >= _MAX_RESPONSES_PER_TURN:
        return SwarmTtsNextTurnResponse(agentName=name, content=full_text, stateToken=None, done=True)

    new_token = _encode_state(body.sessionId, response_count)
    return SwarmTtsNextTurnResponse(agentName=name, content=full_text, stateToken=new_token, done=False)
