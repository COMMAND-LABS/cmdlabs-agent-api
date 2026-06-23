"""
Server-Sent Events (SSE) helpers for agent completion.

Provides consistent formatting for SSE events sent to the client.
"""
import json
from typing import Any


def sse_event(
    event: str,
    data: Any | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
) -> str:
    """
    Create a JSON-encoded SSE event.

    Args:
        event: The event type (e.g., "on_chain_start", "on_chat_model_stream")
        data: Optional data payload for the event
        tool_calls: Optional list of tool calls to include (used only on on_chain_end)
        run_id: Optional LangChain run ID for correlating on_tool_start / on_tool_end pairs

    Returns:
        JSON string ready to be yielded in a StreamingResponse
    """
    payload: dict[str, Any] = {"event": event}

    if data is not None:
        payload["data"] = data

    if run_id:
        payload["run_id"] = run_id

    if tool_calls:  # only include when non-empty
        payload["toolCalls"] = tool_calls

    return json.dumps(payload, separators=(',', ':'))


def sse_error(error: str, message: str) -> str:
    """
    Create a JSON-encoded SSE error event.

    Args:
        error: Short error type/code
        message: Human-readable error message

    Returns:
        JSON string for an error event
    """
    return json.dumps({
        "event": "error",
        "data": {
            "error": error,
            "message": message
        }
    }, separators=(',', ':'))


# Common event types as constants for consistency
class EventType:
    """SSE event type constants."""
    CHAIN_START = "on_chain_start"
    CHAIN_END = "on_chain_end"
    CHAT_MODEL_START = "on_chat_model_start"
    CHAT_MODEL_STREAM = "on_chat_model_stream"
    CHAT_MODEL_END = "on_chat_model_end"
    TOOL_START = "on_tool_start"
    TOOL_END = "on_tool_end"
    ERROR = "error"
    # Emitted when a HITL-gated tool queues an action for human review.
    # Payload: {approval_id, tool_type, preview: {to_email, subject, body}}
    TOOL_APPROVAL_REQUIRED = "tool_approval_required"
