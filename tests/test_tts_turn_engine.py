import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.multi_agent.session_logger import SessionLogger
from src.routers.swarms.turn_engine import execute_turn, stream_turn
from src.routers.swarms.turn_state import decode_and_validate_turn_state, encode_turn_state
from src.routers.swarms.types import (
    AgentDefinition,
    ConversationEntry,
    PreparedTurnContext,
    TurnState,
)


def _build_context(tmp_path, *, response_count: int = 0) -> PreparedTurnContext:
    agent = AgentDefinition(
        name="Jesus",
        description="no description",
        model="gpt-5-mini",
        system_prompt="You are Jesus.",
    )
    return PreparedTurnContext(
        db=object(),
        provider="openai",
        api_key="test-key",
        supervisor_model="gpt-5-mini",
        supervisor_prompt=None,
        agent_definitions={"Jesus": agent},
        agent_list=[{"name": "Jesus", "description": "no description"}],
        history=[ConversationEntry(role="user", content="Hello Jesus")],
        chat_session_id=1,
        state=TurnState(
            session_id="11111111-1111-1111-1111-111111111111",
            response_count=response_count,
            last_message_id=10,
            swarm_hash="swarm-hash",
        ),
        session_logger=SessionLogger("11111111-1111-1111-1111-111111111111", log_dir=str(tmp_path)),
    )


def test_execute_turn_stops_on_natural_pause(tmp_path, monkeypatch):
    async def fake_complete_request(*, provider, model, api_key, request, session_logger=None):
        assert request.label == "router"
        return '{"next":[],"reason":"Natural pause after a greeting."}'

    monkeypatch.setattr("src.routers.swarms.turn_engine.complete_request", fake_complete_request)

    result = asyncio.run(execute_turn(_build_context(tmp_path)))

    assert result.done is True
    assert result.agent_name == ""
    assert result.content == ""
    assert result.state_token is None
    assert result.route_reason == "Natural pause after a greeting."


def test_execute_and_stream_turn_match(tmp_path, monkeypatch):
    async def fake_complete_request(*, provider, model, api_key, request, session_logger=None):
        if request.label == "router":
            return '{"next":["Jesus"],"reason":"The user directly addressed Jesus."}'
        if request.label == "agent:Jesus":
            return "Hello there."
        raise AssertionError(f"Unexpected request label: {request.label}")

    async def fake_stream_request(*, provider, model, api_key, request, session_logger=None):
        assert request.label == "agent:Jesus"
        for chunk in ("Hello ", "there."):
            yield chunk

    def fake_persist_ai_message(db, *, chat_session_id, content, agent_name=None):
        assert chat_session_id == 1
        assert agent_name == "Jesus"
        return SimpleNamespace(id=11)

    monkeypatch.setattr("src.routers.swarms.turn_engine.complete_request", fake_complete_request)
    monkeypatch.setattr("src.routers.swarms.turn_engine.stream_request", fake_stream_request)
    monkeypatch.setattr("src.routers.swarms.turn_engine.persist_ai_message", fake_persist_ai_message)

    json_result = asyncio.run(execute_turn(_build_context(tmp_path / "json")))
    stream_events = asyncio.run(_collect_stream_events(_build_context(tmp_path / "stream")))

    assert [event.event for event in stream_events] == [
        "swarm_agent_start",
        "swarm_chat_model_stream",
        "swarm_chat_model_stream",
        "swarm_agent_end",
        "tts_turn_result",
    ]
    final_event = stream_events[-1]
    assert final_event.data["agentName"] == json_result.agent_name
    assert final_event.data["content"] == json_result.content
    assert final_event.data["stateToken"] == json_result.state_token
    assert final_event.data["done"] == json_result.done
    assert final_event.data["routeReason"] == json_result.route_reason


def test_execute_turn_blocks_same_speaker_repeat(tmp_path, monkeypatch):
    async def fake_complete_request(*, provider, model, api_key, request, session_logger=None):
        assert request.label == "router"
        return '{"next":["Jesus"],"reason":"Jesus should keep talking."}'

    monkeypatch.setattr("src.routers.swarms.turn_engine.complete_request", fake_complete_request)

    context = _build_context(tmp_path)
    context.history = [
        ConversationEntry(role="user", content="Hey Jesus"),
        ConversationEntry(role="assistant", content="Hello! How are you?", agent_name="Jesus"),
    ]

    result = asyncio.run(execute_turn(context))
    assert result.done is True
    assert result.agent_name == ""
    assert result.route_reason == "Suppressed same-speaker repeat."


def test_signed_turn_state_rejects_stale_counts():
    state = TurnState(
        session_id="11111111-1111-1111-1111-111111111111",
        response_count=1,
        last_message_id=44,
        swarm_hash="swarm-hash",
    )
    token = encode_turn_state(state)

    validated = decode_and_validate_turn_state(
        token,
        expected_session_id=state.session_id,
        expected_last_message_id=state.last_message_id,
        expected_swarm_hash=state.swarm_hash,
        expected_response_count=state.response_count,
    )
    assert validated == state

    with pytest.raises(HTTPException):
        decode_and_validate_turn_state(
            token,
            expected_session_id=state.session_id,
            expected_last_message_id=state.last_message_id,
            expected_swarm_hash=state.swarm_hash,
            expected_response_count=state.response_count + 1,
        )


async def _collect_stream_events(context: PreparedTurnContext):
    events = []
    async for event in stream_turn(context):
        events.append(event)
    return events
