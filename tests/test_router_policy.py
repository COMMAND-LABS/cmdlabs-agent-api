"""Tests for the swarm router policy (prompt building and decision parsing)."""

from src.routers.swarms.policy import (
    parse_router_decision,
    sanitize_name,
    build_agent_definitions,
    build_agent_list,
)
from src.routers.swarms.langgraph_schemas import (
    LanggraphSwarmConfigInput,
    LanggraphSupervisorInput,
    LanggraphWorkerInput,
)


def test_parse_valid_decision():
    d = parse_router_decision('{"next":["Alice"],"reason":"Alice is relevant"}', ["Alice", "Bob"])
    assert d.next_speakers == ["Alice"]
    assert d.reason == "Alice is relevant"


def test_parse_empty_next():
    d = parse_router_decision('{"next":[],"reason":"No one should speak"}', ["Alice"])
    assert d.next_speakers == []


def test_parse_filters_invalid_names():
    d = parse_router_decision('{"next":["Eve"],"reason":"test"}', ["Alice", "Bob"])
    assert d.next_speakers == []


def test_parse_malformed_json():
    d = parse_router_decision("not json at all", ["Alice"])
    assert d.next_speakers == []


def test_parse_bare_list():
    d = parse_router_decision('["Alice"]', ["Alice", "Bob"])
    assert d.next_speakers == ["Alice"]


def test_sanitize_name():
    assert sanitize_name("Dr. Smith") == "Dr._Smith"
    assert sanitize_name("simple") == "simple"


def test_build_agent_definitions():
    swarm = LanggraphSwarmConfigInput(
        supervisor=LanggraphSupervisorInput(name="sup"),
        workers=[
            LanggraphWorkerInput(agentName="Alice", systemPrompt="You are Alice."),
            LanggraphWorkerInput(agentName="Bob", agentDescription="A helpful bot."),
        ],
    )
    defs = build_agent_definitions(swarm)
    assert "Alice" in defs
    assert "Bob" in defs
    assert "Alice" in defs["Alice"].system_prompt


def test_build_agent_list():
    swarm = LanggraphSwarmConfigInput(
        supervisor=LanggraphSupervisorInput(name="sup"),
        workers=[LanggraphWorkerInput(agentName="A", agentDescription="desc")],
    )
    defs = build_agent_definitions(swarm)
    agent_list = build_agent_list(defs)
    assert agent_list == [{"name": "A", "description": "desc"}]
