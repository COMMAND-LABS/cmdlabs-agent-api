"""
Multi-agent group chat completion endpoint (SSE streaming).

Simple loop:
  1. Router picks who should speak next (given full shared history).
  2. That agent streams its response.
  3. Repeat until router returns [] or 5 agent responses reached.

All parties see the same shared history — no synthetic prompts are injected.
"""

import json
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.deps import db_dependency, auth_dependency
from src.routers.swarms.langgraph_schemas import LanggraphSwarmCompletionRequest
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.multi_agent import route_next_speaker, stream_agent
from src.multi_agent.session_logger import SessionLogger
from src.routers.swarms.policy import build_agent_definitions, build_agent_list
from src.routers.swarms.repository import (
    DEFAULT_PROVIDER,
    build_account_id,
    build_history,
    db_retry_once,
    load_or_create_chat_session,
    load_session_messages,
    persist_ai_message,
    persist_user_message,
    resolve_api_key,
)

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

_MAX_RESPONSES_PER_TURN = 5


def _sse(event: str, data: dict | None = None) -> str:
    return f"event: {event}\ndata: {json.dumps(data or {}, separators=(',', ':'))}\n\n"


def _sse_error(error: str, message: str) -> str:
    return _sse("error", {"error": error, "message": message})


def _create_llm(provider: str, model: str, api_key: str, *, streaming: bool):
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=api_key, streaming=streaming, temperature=0.7)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=api_key, streaming=streaming, temperature=0.7)
    raise ValueError(f"Unsupported provider: {provider}")


async def _generator(
    request_body: LanggraphSwarmCompletionRequest,
    db,
    auth: dict,
) -> AsyncGenerator[str, None]:
    slog: SessionLogger | None = None
    try:
        account_id = build_account_id(auth)

        # --- resolve API key ---
        provider = DEFAULT_PROVIDER
        try:
            api_key = resolve_api_key(db, account_id=account_id, provider=provider)
        except Exception as exc:
            message = exc.detail if isinstance(exc, Exception) and hasattr(exc, "detail") else str(exc)
            yield _sse_error("API key required", message)
            return

        # --- build agent configs ---
        sw = request_body.swarm
        agent_definitions = build_agent_definitions(sw)
        agent_configs: dict[str, dict] = {}
        agent_list = build_agent_list(agent_definitions)
        for agent in agent_definitions.values():
            agent_configs[agent.name] = {
                "system_prompt": agent.system_prompt,
                "llm": _create_llm(provider, agent.model, api_key, streaming=True),
            }

        router_llm = _create_llm(provider, sw.supervisor.modelName, api_key, streaming=False)

        # --- session & history ---
        session = load_or_create_chat_session(
            db,
            session_id=request_body.sessionId,
            account_id=account_id,
            title="Group agent chat",
        )
        db_messages = load_session_messages(db, chat_session_id=session.id)
        history_entries = build_history(db_messages)
        history = [
            {"role": entry.role, "content": entry.content, "agent_name": entry.agent_name}
            for entry in history_entries
        ]

        chat_session_id = session.id
        prompt = request_body.prompt
        slog = SessionLogger(request_body.sessionId)

        # --- persist user message ---
        persist_user_message(db, chat_session_id=chat_session_id, prompt=prompt)

        history.append({"role": "user", "content": prompt})

        # --- main loop: route → speak → persist → repeat ---
        yield _sse("swarm_run_start")

        all_outputs: list[dict] = []
        response_count = 0

        while response_count < _MAX_RESPONSES_PER_TURN:
            speakers = await route_next_speaker(history, agent_list, router_llm, session_logger=slog)
            slog.log_route(history[-1].get("content", "")[:80], speakers)

            if not speakers:
                break

            name = speakers[0]
            cfg = agent_configs.get(name)
            if not cfg:
                break

            yield _sse("swarm_agent_start", {"agentName": name})
            slog.log_agent_start(name, len(history), history[-1].get("content", "")[:80])
            t0 = SessionLogger.timer()

            full_text = ""
            async for token in stream_agent(
                system_prompt=cfg["system_prompt"],
                history=history,
                llm=cfg["llm"],
                agent_name=name,
                session_logger=slog,
            ):
                full_text += token
                yield _sse("swarm_chat_model_stream", {"agentName": name, "data": token})

            yield _sse("swarm_agent_end", {"agentName": name})
            slog.log_agent_end(name, full_text, SessionLogger.timer() - t0)

            history.append({"role": "assistant", "content": full_text, "agent_name": name})
            all_outputs.append({"agentName": name, "content": full_text})

            persist_ai_message(db, chat_session_id=chat_session_id, content=full_text, agent_name=name)

            response_count += 1

        yield _sse("swarm_run_end", {o["agentName"]: o["content"] for o in all_outputs})

    except Exception as e:
        import traceback
        traceback.print_exc()
        if slog:
            try:
                slog.log_error("generator", e)
            except Exception:
                pass
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
