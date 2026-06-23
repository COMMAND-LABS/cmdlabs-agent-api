"""TTS next-turn endpoints for JSON and SSE transport."""

import contextlib
import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.deps import auth_dependency, db_dependency
from src.multi_agent.session_logger import SessionLogger
from src.ratelimit import limiter
from src.routers.swarms.langgraph_schemas import (
    SwarmTtsNextTurnRequest,
    SwarmTtsNextTurnResponse,
)
from src.routers.swarms.turn_context import prepare_tts_turn_context
from src.routers.swarms.turn_engine import execute_turn, stream_turn

logger = logging.getLogger(__name__)

router = APIRouter()


def _sse(event: str, data: dict | None = None) -> str:
    body = json.dumps(data or {}, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n"


def _sse_error(error: str, message: str) -> str:
    return _sse("error", {"error": error, "message": message})


async def _stream_tts_turn_generator(
    *,
    body: SwarmTtsNextTurnRequest,
    db,
    auth: dict,
) -> AsyncGenerator[str, None]:
    slog: SessionLogger | None = None
    try:
        context = prepare_tts_turn_context(body=body, db=db, auth=auth)
        slog = context.session_logger
        async for event in stream_turn(context):
            yield _sse(event.event, event.data)
    except HTTPException as e:
        if slog:
            with contextlib.suppress(Exception):
                slog.log_error("tts_next_turn_stream", e)
        detail = e.detail if isinstance(e.detail, str) else json.dumps(e.detail)
        yield _sse_error(f"http_{e.status_code}", detail)
    except Exception as e:
        logger.exception(f"[TTS NEXT TURN] Unhandled error during turn stream: {e!s}")
        if slog:
            with contextlib.suppress(Exception):
                slog.log_error("tts_next_turn_stream", e)
        yield _sse_error("internal_server_error", str(e))


@router.post("/tts/next-turn", response_model=SwarmTtsNextTurnResponse)
@limiter.limit("200/minute")
async def swarm_tts_next_turn(
    request: Request,
    body: SwarmTtsNextTurnRequest,
    db: db_dependency,
    auth: auth_dependency,
) -> SwarmTtsNextTurnResponse:
    """Return one agent's turn, then route to decide if someone else should speak."""
    context = prepare_tts_turn_context(body=body, db=db, auth=auth)
    result = await execute_turn(context)
    return SwarmTtsNextTurnResponse(
        agentName=result.agent_name,
        content=result.content,
        stateToken=result.state_token,
        done=result.done,
    )


@router.post("/tts/next-turn/stream")
@limiter.limit("200/minute")
async def swarm_tts_next_turn_stream(
    request: Request,
    body: SwarmTtsNextTurnRequest,
    db: db_dependency,
    auth: auth_dependency,
) -> StreamingResponse:
    """Stream one TTS agent turn with SSE token events."""
    return StreamingResponse(
        _stream_tts_turn_generator(
            body=body,
            db=db,
            auth=auth,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
