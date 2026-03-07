"""
Server-Sent Events (SSE) helpers for agent completion.

Provides consistent formatting for SSE events sent to the client.
"""
import json
from typing import Any, Optional, List, Dict


def sse_event(
    event: str,
    data: Optional[Any] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Create a JSON-encoded SSE event.
    
    Args:
        event: The event type (e.g., "on_chain_start", "on_chat_model_stream")
        data: Optional data payload for the event
        tool_calls: Optional list of tool calls to include
        
    Returns:
        JSON string ready to be yielded in a StreamingResponse
    """
    payload: Dict[str, Any] = {"event": event}
    
    if data is not None:
        payload["data"] = data
    
    if tool_calls is not None:
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
    # Hierarchical swarm event types
    SWARM_RUN_START = "swarm_run_start"
    SWARM_DIRECTOR_START = "swarm_director_start"
    SWARM_DIRECTOR_DONE = "swarm_director_done"
    SWARM_AGENT_START = "swarm_agent_start"
    SWARM_CHAT_MODEL_STREAM = "swarm_chat_model_stream"
    SWARM_AGENT_END = "swarm_agent_end"
    SWARM_LOOP_END = "swarm_loop_end"
    SWARM_RUN_END = "swarm_run_end"


def sse_swarm_event(
    event: str,
    *,
    agent_name: Optional[str] = None,
    data: Optional[Any] = None,
    loop_index: Optional[int] = None,
    **extra: Any
) -> str:
    """
    Create a JSON-encoded SSE event for hierarchical swarm stream.
    Payload can include agentName, data, loopIndex for UI to identify speaker and state.
    """
    payload: Dict[str, Any] = {"event": event}
    if agent_name is not None:
        payload["agentName"] = agent_name
    if data is not None:
        payload["data"] = data
    if loop_index is not None:
        payload["loopIndex"] = loop_index
    payload.update(extra)
    return json.dumps(payload, separators=(',', ':'))
