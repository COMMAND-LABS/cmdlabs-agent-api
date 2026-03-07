"""
LangGraph supervisor swarm completion endpoint.

Runs the langgraph-supervisor graph via ainvoke, then streams the
worker outputs as SSE events matching the swarm protocol.
"""
import json
import logging
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

from src.langgraph_swarm import run_langgraph_swarm, _to_node_name

logger = logging.getLogger("completion-api")
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
    return json.dumps(payload, separators=(",", ":"))


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
        return ChatOpenAI(model=model, api_key=api_key, streaming=streaming, temperature=0.7, stream_usage=True)
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

        sw = request_body.swarm
        supervisor_llm = _create_llm(provider, sw.supervisor.modelName, api_key, streaming=False)

        worker_configs = []
        worker_llms = {}
        for w in sw.workers:
            node_name = _to_node_name(w.agentName)
            worker_configs.append({
                "node_name": node_name,
                "display_name": w.agentName,
                "system_prompt": w.systemPrompt or f"You are {w.agentName}.",
            })
            worker_llms[node_name] = _create_llm(provider, w.modelName, api_key, streaming=False)

        node_names = [wc["node_name"] for wc in worker_configs]
        supervisor_prompt = sw.supervisor.systemPrompt or (
            "You are a team supervisor managing the following agents: "
            + ", ".join(node_names) + ". "
            "You MUST ALWAYS delegate to at least one agent using the transfer tools. "
            "NEVER respond directly to the user yourself. "
            "For every user message, choose the most appropriate agent and hand off to them."
        )

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
                history.append({"role": role, "content": md["content"]})

        chat_session_id = session.id
        prompt = request_body.prompt
        db.close()

        try:
            _store_user_msg(chat_session_id, prompt)
        except Exception:
            pass

        # Run the graph
        print("[LANGGRAPH] About to yield swarm_run_start", flush=True)
        yield _sse({"event": "swarm_run_start"})
        print("[LANGGRAPH] Yielded swarm_run_start, calling ainvoke...", flush=True)

        try:
            result = await run_langgraph_swarm(
                prompt=prompt,
                history=history,
                supervisor_llm=supervisor_llm,
                supervisor_prompt=supervisor_prompt,
                worker_configs=worker_configs,
                worker_llms=worker_llms,
                output_mode=sw.outputMode,
            )
            print(f"[LANGGRAPH] ainvoke complete, agents={list(result['agent_outputs'].keys())}", flush=True)
        except Exception as invoke_err:
            print(f"[LANGGRAPH] ainvoke FAILED: {invoke_err}", flush=True)
            import traceback
            traceback.print_exc()
            yield _sse_error("Graph execution failed", str(invoke_err))
            return

        agent_outputs = result["agent_outputs"]

        for agent_name, content in agent_outputs.items():
            yield _sse({"event": "swarm_agent_start", "agentName": agent_name})
            yield _sse({"event": "swarm_chat_model_stream", "agentName": agent_name, "data": content})
            yield _sse({"event": "swarm_agent_end", "agentName": agent_name})

        yield _sse({"event": "swarm_run_end", "data": agent_outputs})
        print(f"[LANGGRAPH] All events yielded", flush=True)

        # Persist AI messages
        try:
            for agent_name, content in agent_outputs.items():
                if content:
                    _store_ai_msg(chat_session_id, content, agent_name=agent_name)
        except Exception:
            pass

    except GeneratorExit:
        print("[LANGGRAPH] Generator cancelled (client disconnected)", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[LANGGRAPH] Outer exception: {e}", flush=True)
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
    logger.info(
        "langgraph/completion started sessionId=%s prompt_len=%d workers=%d",
        request_body.sessionId,
        len(request_body.prompt or ""),
        len(request_body.swarm.workers or []),
    )
    return StreamingResponse(
        _langgraph_generator(
            request_body=request_body,
            db=db,
            auth=auth,
        ),
        media_type="text/event-stream",
    )
