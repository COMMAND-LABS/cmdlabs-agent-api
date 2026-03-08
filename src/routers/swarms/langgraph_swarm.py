"""
LangGraph supervisor swarm completion endpoint.

Streams the same SSE event protocol as the in-house swarm but backed by
langgraph-supervisor's create_supervisor + create_react_agent.
"""
import asyncio
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

from src.langgraph_swarm import stream_langgraph_swarm, _to_node_name

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


async def _langgraph_generator(
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

        # --- build LLMs ---
        sw = request_body.swarm
        supervisor_llm = _create_llm(provider, sw.supervisor.modelName, api_key, streaming=False)

        all_display_names = [w.agentName for w in sw.workers]
        room_members = ", ".join(all_display_names)

        worker_configs = []
        worker_llms = {}
        for w in sw.workers:
            node_name = _to_node_name(w.agentName)
            base_prompt = w.systemPrompt or f"You are {w.agentName}."
            full_prompt = (
                f"{base_prompt}\n\n"
                f"You are in a room with a human and: {room_members}. "
                "Be yourself. Keep it natural and concise."
            )
            worker_configs.append({
                "node_name": node_name,
                "display_name": w.agentName,
                "agent_description": w.agentDescription,
                "system_prompt": full_prompt,
            })
            worker_llms[node_name] = _create_llm(provider, w.modelName, api_key, streaming=True)

        supervisor_prompt = sw.supervisor.systemPrompt or (
            f"You are in a chat room with a human and: {room_members}. "
            "When the human says something, hand off to whoever should respond. "
            "Always hand off to someone — never skip. "
            "After they reply, say 'done'."
        )

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
                title="LangGraph swarm chat",
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

        history = []
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

        # --- stream events from LangGraph ---
        agent_outputs: dict[str, str] = {}
        async for evt in stream_langgraph_swarm(
            prompt=prompt,
            history=history,
            supervisor_llm=supervisor_llm,
            supervisor_prompt=supervisor_prompt,
            worker_configs=worker_configs,
            worker_llms=worker_llms,
            output_mode=sw.outputMode,
        ):
            event_type = evt.get("event")

            if event_type == "swarm_chat_model_stream":
                agent_name = evt.get("agentName", "")
                chunk = evt.get("data", "")
                if agent_name and chunk:
                    agent_outputs[agent_name] = agent_outputs.get(agent_name, "") + chunk
                yield _sse(evt)

            elif event_type == "error":
                yield _sse_error("Swarm error", evt.get("data", "Unknown error"))
                return

            elif event_type in ("swarm_agent_end", "swarm_run_end"):
                await asyncio.sleep(0.05)
                yield _sse(evt)

            else:
                yield _sse(evt)

        # --- persist AI messages ---
        for agent_name, content in agent_outputs.items():
            print(f"[LANGGRAPH] final output [{agent_name}]: {content!r}", flush=True)

        try:
            for agent_name, content in agent_outputs.items():
                if content:
                    _store_ai_msg(chat_session_id, content, agent_name=agent_name)
        except Exception:
            pass

        # Flush padding — ensures reverse proxies deliver the last chunk
        yield "\n"

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
    """Stream LangGraph supervisor swarm completion with per-agent SSE events."""
    return StreamingResponse(
        _langgraph_generator(
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
