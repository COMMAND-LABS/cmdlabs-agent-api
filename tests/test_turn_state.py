"""Tests for swarm turn state encoding/validation."""

import pytest
from fastapi import HTTPException

from src.routers.swarms.turn_state import (
    compute_authoritative_response_count,
    compute_swarm_hash,
    decode_and_validate_turn_state,
    encode_turn_state,
)
from src.routers.swarms.types import ConversationEntry, TurnState


def _make_state(**overrides):
    defaults = {
        "session_id": "aaaa-bbbb",
        "response_count": 2,
        "last_message_id": 50,
        "swarm_hash": "abc123",
    }
    defaults.update(overrides)
    return TurnState(**defaults)


def test_roundtrip():
    state = _make_state()
    token = encode_turn_state(state)
    decoded = decode_and_validate_turn_state(
        token,
        expected_session_id=state.session_id,
        expected_last_message_id=state.last_message_id,
        expected_swarm_hash=state.swarm_hash,
        expected_response_count=state.response_count,
    )
    assert decoded == state


def test_tampered_signature_rejected():
    token = encode_turn_state(_make_state())
    tampered = token[:-4] + "XXXX"
    with pytest.raises(HTTPException) as exc_info:
        decode_and_validate_turn_state(
            tampered,
            expected_session_id="aaaa-bbbb",
            expected_last_message_id=50,
            expected_swarm_hash="abc123",
            expected_response_count=2,
        )
    assert exc_info.value.status_code == 400


def test_wrong_session_id_rejected():
    state = _make_state()
    token = encode_turn_state(state)
    with pytest.raises(HTTPException):
        decode_and_validate_turn_state(
            token,
            expected_session_id="wrong-session",
            expected_last_message_id=state.last_message_id,
            expected_swarm_hash=state.swarm_hash,
            expected_response_count=state.response_count,
        )


def test_stale_message_id_is_409():
    state = _make_state()
    token = encode_turn_state(state)
    with pytest.raises(HTTPException) as exc_info:
        decode_and_validate_turn_state(
            token,
            expected_session_id=state.session_id,
            expected_last_message_id=999,
            expected_swarm_hash=state.swarm_hash,
            expected_response_count=state.response_count,
        )
    assert exc_info.value.status_code == 409


def test_authoritative_response_count():
    history = [
        ConversationEntry(role="user", content="hi"),
        ConversationEntry(role="assistant", content="hello", agent_name="A"),
        ConversationEntry(role="assistant", content="hey", agent_name="B"),
    ]
    assert compute_authoritative_response_count(history) == 2


def test_authoritative_count_resets_on_user():
    history = [
        ConversationEntry(role="user", content="first"),
        ConversationEntry(role="assistant", content="reply"),
        ConversationEntry(role="user", content="second"),
        ConversationEntry(role="assistant", content="reply2"),
    ]
    assert compute_authoritative_response_count(history) == 1


def test_swarm_hash_deterministic():
    from src.routers.swarms.langgraph_schemas import (
        LanggraphSupervisorInput,
        LanggraphSwarmConfigInput,
        LanggraphWorkerInput,
    )
    swarm = LanggraphSwarmConfigInput(
        supervisor=LanggraphSupervisorInput(name="sup"),
        workers=[LanggraphWorkerInput(agentName="A")],
    )
    h1 = compute_swarm_hash(swarm)
    h2 = compute_swarm_hash(swarm)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex
