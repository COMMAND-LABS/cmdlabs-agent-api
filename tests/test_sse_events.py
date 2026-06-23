"""Tests for SSE event formatting."""

import json

from src.routers.agents.helpers.sse_events import EventType, sse_error, sse_event


def test_sse_event_basic():
    result = json.loads(sse_event("on_chain_start"))
    assert result == {"event": "on_chain_start"}


def test_sse_event_with_data():
    result = json.loads(sse_event("on_chat_model_stream", data="hello"))
    assert result["event"] == "on_chat_model_stream"
    assert result["data"] == "hello"


def test_sse_event_with_tool_calls():
    calls = [{"toolType": "vectorSearch", "toolName": "search"}]
    result = json.loads(sse_event("on_chain_end", data="done", tool_calls=calls))
    assert result["toolCalls"] == calls


def test_sse_event_omits_empty_tool_calls():
    result = json.loads(sse_event("on_chain_end", tool_calls=[]))
    assert "toolCalls" not in result


def test_sse_event_with_run_id():
    result = json.loads(sse_event("on_tool_start", run_id="abc-123"))
    assert result["run_id"] == "abc-123"


def test_sse_error():
    result = json.loads(sse_error("Not found", "Agent does not exist"))
    assert result["event"] == "error"
    assert result["data"]["error"] == "Not found"
    assert result["data"]["message"] == "Agent does not exist"


def test_event_type_constants():
    assert EventType.CHAIN_START == "on_chain_start"
    assert EventType.TOOL_APPROVAL_REQUIRED == "tool_approval_required"
