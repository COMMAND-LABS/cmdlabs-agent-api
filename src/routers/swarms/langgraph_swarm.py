"""
Multi-agent group chat completion endpoint.

Uses a simple route-and-stream pattern:
  1. A fast router LLM call picks which agent(s) should respond.
  2. Each selected agent streams tokens directly via llm.astream().

No graph framework, no background tasks, no queues.
"""

import json
import time
import uuid
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import OperationalError

from src.deps import db_dependency, auth_dependency
from src.db.database import SessionLocal
from src.db.models import ChatSession, ChatMessage, Credential
from src.routers.credentials.encryption import get_credential_value
from src.routers.swarms.langgraph_schemas import LanggraphSwarmCompletionRequest
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.multi_agent import route_message, stream_agent

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

_DEFAULT_PROVIDER = "openai"


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
        time.sleep(0.5)
        return fn()


def _sse(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":")) + "\n"


def _sse_error(error: str, message: str) -> str:
    return _sse({"event": "error", "data": {"error": error, "message": message}})


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
        return ChatOpenAI(model=model, api_key=api_key, streaming=streaming, temperature=0.7)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=api_key, streaming=streaming, temperature=0.7)
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


async def _generator(
    request_body: LanggraphSwarmCompletionRequest,
    db,
    auth: dict,
) -> AsyncGenerator[str, None]:
    try:
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
                yield _sse_error("API key required", f"Please add your {provider.title()} API key in account settings.")
                return
            try:
                api_key = get_credential_value(cred, "api_key")
            except Exception as e:
                yield _sse_error("Failed to retrieve API key", str(e))
                return
        else:
            api_key = ""

        # --- build agent configs ---
        sw = request_body.swarm
        all_display_names = [w.agentName for w in sw.workers]
        room_members = ", ".join(all_display_names)

        agent_configs: dict[str, dict] = {}
        agent_list: list[dict[str, str]] = []
        for w in sw.workers:
            base_prompt = w.systemPrompt or f"You are {w.agentName}."
            full_prompt = (
                f"{base_prompt}\n\n"
                f"You are in a group conversation with a human and: {room_members}.\n"
                "- Speak naturally as yourself. Only say your own words.\n"
                "- Do NOT end every message with a question. Sometimes just share "
                "a thought, react, or make a statement.\n"
                "- Be direct. If someone asks who you were talking to, answer honestly.\n"
                "- Avoid filler like \"I appreciate your perspective\" — just respond.\n"
                "- Keep it concise. A few sentences is usually enough."
            )
            agent_configs[w.agentName] = {
                "system_prompt": full_prompt,
                "llm": _create_llm(provider, w.modelName, api_key, streaming=True),
            }
            agent_list.append({
                "name": w.agentName,
                "description": w.agentDescription or "",
            })

        router_llm = _create_llm(provider, sw.supervisor.modelName, api_key, streaming=False)

        # --- session & history ---
        try:
            session_uuid = uuid.UUID(request_body.sessionId)
        except ValueError:
            yield _sse_error("Invalid sessionId format", "The sessionId must be a valid UUID.")
            return

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
                title="Group agent chat",
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

        history: list[dict[str, str]] = []
        for msg in db_messages:
            md = msg.message
            if isinstance(md, dict) and "role" in md and "content" in md:
                role = "user" if md["role"] == "human" else "assistant"
                content = md["content"]
                if role == "assistant":
                    agent_name = md.get("agentName")
                    if agent_name:
                        content = f"[{agent_name}]: {content}"
                history.append({"role": role, "content": content})

        chat_session_id = session.id
        prompt = request_body.prompt
        db.close()

        # --- persist user message ---
        try:
            _store_user_msg(chat_session_id, prompt)
        except Exception:
            pass

        # --- Step 1: Route — pick who should speak ---
        speakers = await route_message(prompt, history, agent_list, router_llm)
        print(f"[MULTI-AGENT] routed to: {speakers}", flush=True)

        # --- Step 2: Stream each speaker sequentially ---
        yield _sse({"event": "swarm_run_start"})

        running_history = list(history)
        multi = len(speakers) > 1
        agent_outputs: dict[str, str] = {}
        for i, name in enumerate(speakers):
            cfg = agent_configs.get(name)
            if not cfg:
                continue

            if not multi or i == 0:
                effective_prompt = prompt
            else:
                effective_prompt = "Continue the conversation."

            yield _sse({"event": "swarm_agent_start", "agentName": name})

            full_text = ""
            async for token in stream_agent(
                system_prompt=cfg["system_prompt"],
                history=running_history,
                prompt=effective_prompt,
                llm=cfg["llm"],
                agent_name=name,
            ):
                full_text += token
                yield _sse({"event": "swarm_chat_model_stream", "agentName": name, "data": token})

            yield _sse({"event": "swarm_agent_end", "agentName": name})
            agent_outputs[name] = full_text
            running_history.append({"role": "assistant", "content": f"[{name}]: {full_text}"})
            print(f"[MULTI-AGENT] {name}: {full_text!r}", flush=True)

        yield _sse({"event": "swarm_run_end", "data": agent_outputs})

        # --- persist AI messages ---
        try:
            for agent_name, content in agent_outputs.items():
                if content:
                    _store_ai_msg(chat_session_id, content, agent_name=agent_name)
        except Exception:
            pass

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield _sse_error("Internal server error", str(e))


@router.post("/langgraph/completion")
@limiter.limit("200/minute")
async def langgraph_swarm_completion(
    request: Request,
    request_body: LanggraphSwarmCompletionRequest,
    db: db_dependency,
    auth: auth_dependency,
) -> StreamingResponse:
    """Stream multi-agent group chat completion with per-agent SSE events."""
    return StreamingResponse(
        _generator(
            request_body=request_body,
            db=db,
            auth=auth,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
