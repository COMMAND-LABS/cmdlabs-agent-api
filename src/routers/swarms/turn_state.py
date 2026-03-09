from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from typing import Optional

from fastapi import HTTPException

from src.routers.swarms.langgraph_schemas import LanggraphSwarmConfigInput
from src.routers.swarms.types import ConversationEntry, TurnState


def compute_swarm_hash(swarm: LanggraphSwarmConfigInput) -> str:
    payload = json.dumps(swarm.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def compute_authoritative_response_count(history: list[ConversationEntry]) -> int:
    count = 0
    for entry in reversed(history):
        if entry.role == "assistant":
            count += 1
            continue
        if entry.role == "user":
            break
    return count


def build_turn_state(
    *,
    session_id: str,
    history: list[ConversationEntry],
    last_message_id: Optional[int],
    swarm_hash: str,
) -> TurnState:
    return TurnState(
        session_id=session_id,
        response_count=compute_authoritative_response_count(history),
        last_message_id=last_message_id,
        swarm_hash=swarm_hash,
    )


def encode_turn_state(state: TurnState) -> str:
    payload = {
        "sessionId": state.session_id,
        "responseCount": state.response_count,
        "lastMessageId": state.last_message_id,
        "swarmHash": state.swarm_hash,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    secret = _signing_secret()
    signature = hmac.new(secret, raw, hashlib.sha256).hexdigest()
    token = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    return f"{token}.{signature}"


def decode_and_validate_turn_state(
    token: str,
    *,
    expected_session_id: str,
    expected_last_message_id: Optional[int],
    expected_swarm_hash: str,
    expected_response_count: int,
) -> TurnState:
    try:
        token_payload, signature = token.rsplit(".", 1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid stateToken format.") from exc

    raw = _decode_token_payload(token_payload)
    expected_signature = hmac.new(_signing_secret(), raw, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(status_code=400, detail="Invalid stateToken signature.")

    try:
        payload = json.loads(raw.decode())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid stateToken payload.") from exc

    state = TurnState(
        session_id=str(payload.get("sessionId") or ""),
        response_count=int(payload.get("responseCount") or 0),
        last_message_id=payload.get("lastMessageId"),
        swarm_hash=str(payload.get("swarmHash") or ""),
    )
    if state.session_id != expected_session_id:
        raise HTTPException(status_code=400, detail="stateToken does not match sessionId.")
    if state.swarm_hash != expected_swarm_hash:
        raise HTTPException(status_code=400, detail="stateToken does not match swarm configuration.")
    if state.last_message_id != expected_last_message_id:
        raise HTTPException(status_code=409, detail="stateToken is stale for the current conversation state.")
    if state.response_count != expected_response_count:
        raise HTTPException(status_code=409, detail="stateToken response count is out of sync with conversation state.")
    return state


def _decode_token_payload(token_payload: str) -> bytes:
    padding = "=" * (-len(token_payload) % 4)
    try:
        return base64.urlsafe_b64decode((token_payload + padding).encode())
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid stateToken format.") from exc


def _signing_secret() -> bytes:
    secret = os.getenv("AUTH_SECRET_KEY")
    if not secret:
        raise RuntimeError("AUTH_SECRET_KEY is required for signed state tokens.")
    return secret.encode()
