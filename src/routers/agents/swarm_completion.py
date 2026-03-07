"""
Hierarchical swarm completion endpoint.

Streams SSE events: swarm_run_start, swarm_director_start, swarm_director_done,
swarm_agent_start, swarm_chat_model_stream, swarm_agent_end, swarm_loop_end, swarm_run_end.
"""
import asyncio
import queue
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import OperationalError

from src.deps import db_dependency, auth_dependency
from src.db.database import SessionLocal
from src.db.models import Agent, ChatSession, ChatMessage, Credential
from src.routers.agents.access import load_agent_with_access_check
from src.routers.credentials.encryption import get_credential_value
from src.core.schemas.ChatSessionPrompt import ChatSessionPrompt
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.swarm import run_swarm_streaming, parse_swarm_config
from src.swarm.runner import SWARM_END
from src.routers.agents.helpers import (
    build_swarm_history,
    store_user_message,
    store_ai_message,
    sse_swarm_event,
    sse_error,
    create_llm,
    get_required_credential_type,
)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

# Default provider for swarm (director/workers use OpenAI unless specified)
SWARM_DEFAULT_PROVIDER = "openai"


def _db_retry_once(db, operation_name: str, fn):
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


def _model_config_for(model_name: str, provider: str = SWARM_DEFAULT_PROVIDER) -> dict:
    return {"provider": provider, "model": model_name or "gpt-4o-mini"}


async def _swarm_generator(
    agent_id: int,
    session_id: str,
    prompt: str,
    db,
    auth: dict,
) -> any:
    try:
        account_id = int(auth["id"]) if isinstance(auth["id"], str) else auth["id"]
        agent = _db_retry_once(
            db,
            "load agent",
            lambda: load_agent_with_access_check(db, account_id, agent_id),
        )
        if not agent:
            yield sse_error("Agent not found", "The specified agent was not found or you do not have access.")
            return
        if not agent.config:
            yield sse_error("Invalid agent configuration", "Agent configuration is missing.")
            return

        swarm_config = parse_swarm_config(agent.config)
        if not swarm_config:
            yield sse_error(
                "Not a swarm agent",
                "This agent is not configured as a hierarchical swarm. Set multiAgentArchitecture to hierarchicalSwarm and add swarm.director and swarm.workers.",
            )
            return

        provider = SWARM_DEFAULT_PROVIDER
        required_credential = get_required_credential_type(provider)
        credentials = {}
        if required_credential:
            cred = _db_retry_once(
                db,
                "load credential",
                lambda: db.query(Credential).filter(
                    Credential.account_id == account_id,
                    Credential.service_name == required_credential,
                ).first(),
            )
            if not cred:
                yield sse_error(
                    "API key required",
                    f"Please add your {provider.title()} API key in account settings.",
                )
                return
            try:
                credentials[provider] = get_credential_value(cred, "api_key")
            except Exception as e:
                yield sse_error("Failed to retrieve API key", str(e))
                return

        director_llm, _ = create_llm(
            model_config=_model_config_for(swarm_config.director.model_name),
            credentials=credentials,
            streaming=False,
            temperature=0.7,
        )
        worker_llms = {}
        for w in swarm_config.workers:
            worker_llms[w.agent_name], _ = create_llm(
                model_config=_model_config_for(w.model_name),
                credentials=credentials,
                streaming=True,
                temperature=0.7,
            )

        try:
            session_uuid = uuid.UUID(session_id)
        except ValueError:
            yield sse_error("Invalid sessionId format", "The sessionId must be a valid UUID.")
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
                agent_id=agent_id,
                account_id=account_id,
                title=f"Swarm chat {agent_id}",
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
        history = build_swarm_history(db_messages)
        task = f"User: {prompt}"
        chat_session_id = session.id
        db.close()

        event_queue = queue.Queue()
        current_agent: Optional[str] = None
        agent_buffer: dict[str, str] = {}
        worker_outputs: list[tuple[str, str]] = []

        def run_in_thread():
            run_swarm_streaming(
                task=task,
                history=history,
                swarm_config=swarm_config,
                director_llm=director_llm,
                worker_llms=worker_llms,
                event_queue=event_queue,
            )

        loop = asyncio.get_event_loop()
        task_future = asyncio.ensure_future(loop.run_in_executor(None, run_in_thread))

        try:
            store_user_message(SessionLocal(), chat_session_id, prompt)
        except Exception:
            pass

        while True:
            try:
                item = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: event_queue.get(timeout=0.5),
                )
            except queue.Empty:
                if task_future.done():
                    break
                await asyncio.sleep(0.05)
                continue
            except Exception:
                break

            if not isinstance(item, (list, tuple)) or len(item) < 1:
                continue
            event_type = item[0]
            if event_type == SWARM_END or event_type == "_swarm_end":
                break
            if event_type == "error":
                yield sse_error("Swarm error", item[1] if len(item) > 1 else "Unknown error")
                break
            if event_type == "swarm_run_start":
                yield sse_swarm_event("swarm_run_start")
                continue
            if event_type == "swarm_director_start":
                yield sse_swarm_event("swarm_director_start")
                continue
            if event_type == "swarm_director_done" and len(item) >= 2:
                yield sse_swarm_event("swarm_director_done", data=item[1])
                continue
            if event_type == "stream" and len(item) >= 4:
                _, agent_name, chunk, is_final = item[0], item[1], item[2], item[3]
                if current_agent != agent_name:
                    current_agent = agent_name
                    agent_buffer[agent_name] = ""
                    yield sse_swarm_event("swarm_agent_start", agent_name=agent_name)
                if chunk:
                    agent_buffer[agent_name] = agent_buffer.get(agent_name, "") + chunk
                    yield sse_swarm_event("swarm_chat_model_stream", agent_name=agent_name, data=chunk)
                if is_final:
                    yield sse_swarm_event("swarm_agent_end", agent_name=agent_name)
                    worker_outputs.append((agent_name, agent_buffer.get(agent_name, "")))
                    current_agent = None
                continue
            if event_type == "swarm_loop_end" and len(item) >= 2:
                yield sse_swarm_event("swarm_loop_end", loop_index=item[1])
                continue
            if event_type == "swarm_run_end" and len(item) >= 2:
                yield sse_swarm_event("swarm_run_end", data=item[1])
                continue

        await task_future
        try:
            db = SessionLocal()
            for agent_name, content in worker_outputs:
                if content:
                    store_ai_message(db, chat_session_id, content, agent_name=agent_name)
            db.close()
        except Exception:
            pass

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield sse_error("Internal server error", str(e))


@router.post("/{agent_id}/swarm-completion")
@limiter.limit("200/minute")
async def swarm_completion(
    request: Request,
    agent_id: int,
    request_body: ChatSessionPrompt,
    db: db_dependency,
    auth: auth_dependency,
):
    """Stream hierarchical swarm completion with per-agent SSE events."""
    return StreamingResponse(
        _swarm_generator(
            agent_id=agent_id,
            session_id=request_body.sessionId,
            prompt=request_body.prompt,
            db=db,
            auth=auth,
        ),
        media_type="text/event-stream",
    )
