from __future__ import annotations

from fastapi import HTTPException

from src.multi_agent.session_logger import SessionLogger
from src.routers.swarms.langgraph_schemas import SwarmTtsNextTurnRequest
from src.routers.swarms.policy import build_agent_definitions, build_agent_list
from src.routers.swarms.repository import (
    DEFAULT_PROVIDER,
    build_account_id,
    build_history,
    latest_message_id,
    load_or_create_chat_session,
    load_session_messages,
    persist_user_message,
    resolve_api_key,
)
from src.routers.swarms.turn_state import build_turn_state, compute_swarm_hash, decode_and_validate_turn_state
from src.routers.swarms.types import ConversationEntry, PreparedTurnContext


def prepare_tts_turn_context(
    *,
    body: SwarmTtsNextTurnRequest,
    db,
    auth: dict,
) -> PreparedTurnContext:
    account_id = build_account_id(auth)
    provider = DEFAULT_PROVIDER
    api_key = resolve_api_key(db, account_id=account_id, provider=provider)
    agent_definitions = build_agent_definitions(body.swarm)
    agent_list = build_agent_list(agent_definitions)
    session = load_or_create_chat_session(
        db,
        session_id=body.sessionId,
        account_id=account_id,
        title="Multi-agent TTS chat",
    )

    db_messages = load_session_messages(db, chat_session_id=session.id)
    history = build_history(db_messages)
    current_last_message_id = latest_message_id(db_messages)
    swarm_hash = compute_swarm_hash(body.swarm)
    current_state = build_turn_state(
        session_id=body.sessionId,
        history=history,
        last_message_id=current_last_message_id,
        swarm_hash=swarm_hash,
    )

    if body.stateToken:
        decode_and_validate_turn_state(
            body.stateToken,
            expected_session_id=body.sessionId,
            expected_last_message_id=current_state.last_message_id,
            expected_swarm_hash=current_state.swarm_hash,
            expected_response_count=current_state.response_count,
        )
    else:
        prompt = (body.prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required when stateToken is not provided.")
        persist_user_message(db, chat_session_id=session.id, prompt=prompt)
        history.append(ConversationEntry(role="user", content=prompt))

    # Rebuild state after optional user persistence so the engine always sees the latest state.
    db_messages = load_session_messages(db, chat_session_id=session.id)
    history = build_history(db_messages)
    turn_state = build_turn_state(
        session_id=body.sessionId,
        history=history,
        last_message_id=latest_message_id(db_messages),
        swarm_hash=swarm_hash,
    )

    return PreparedTurnContext(
        db=db,
        provider=provider,
        api_key=api_key,
        supervisor_model=body.swarm.supervisor.modelName,
        supervisor_prompt=body.swarm.supervisor.systemPrompt,
        agent_definitions=agent_definitions,
        agent_list=agent_list,
        history=history,
        chat_session_id=session.id,
        state=turn_state,
        session_logger=SessionLogger(body.sessionId),
    )
