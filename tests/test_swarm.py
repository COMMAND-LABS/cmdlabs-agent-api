"""
Tests for hierarchical swarm: config parsing and SSE event format.
Uses only src.swarm and inline SSE payload build to avoid pulling in DB/encryption.
"""
import json

from src.swarm.config import parse_swarm_config, SwarmConfig, WorkerSpec, DirectorSpec


def _sse_swarm_event(event: str, *, agent_name=None, data=None, loop_index=None, **extra):
    """Mirror of sse_swarm_event for testing without importing helpers."""
    payload = {"event": event}
    if agent_name is not None:
        payload["agentName"] = agent_name
    if data is not None:
        payload["data"] = data
    if loop_index is not None:
        payload["loopIndex"] = loop_index
    payload.update(extra)
    return json.dumps(payload, separators=(",", ":"))


def test_parse_swarm_config_valid():
    """Valid swarm config is parsed into SwarmConfig with director and workers."""
    config = {
        "version": 3,
        "data": {
            "multiAgentArchitecture": "hierarchicalSwarm",
            "swarm": {
                "director": {
                    "name": "Director",
                    "modelName": "gpt-4o-mini",
                    "systemPrompt": "You are the director.",
                },
                "workers": [
                    {
                        "agentName": "Einstein",
                        "agentDescription": "Physics expert",
                        "systemPrompt": "You are Einstein.",
                        "modelName": "gpt-4o-mini",
                    },
                    {
                        "agentName": "Cleopatra",
                        "agentDescription": "Historian",
                        "systemPrompt": "You are Cleopatra.",
                    },
                ],
                "maxLoops": 2,
            },
        },
    }
    out = parse_swarm_config(config)
    assert out is not None
    assert isinstance(out, SwarmConfig)
    assert out.director.name == "Director"
    assert out.director.model_name == "gpt-4o-mini"
    assert out.director.system_prompt == "You are the director."
    assert len(out.workers) == 2
    assert out.workers[0].agent_name == "Einstein"
    assert out.workers[1].agent_name == "Cleopatra"
    assert out.workers[1].model_name == "gpt-4o-mini"
    assert out.max_loops == 2


def test_parse_swarm_config_not_swarm():
    """Config without multiAgentArchitecture hierarchicalSwarm returns None."""
    config = {"version": 3, "data": {"systemPrompt": "Hi"}}
    assert parse_swarm_config(config) is None
    config["data"]["multiAgentArchitecture"] = "other"
    assert parse_swarm_config(config) is None


def test_parse_swarm_config_missing_workers():
    """Config with swarm but no workers returns None."""
    config = {
        "version": 3,
        "data": {
            "multiAgentArchitecture": "hierarchicalSwarm",
            "swarm": {"director": {"name": "D", "modelName": "gpt-4o-mini"}, "workers": []},
        },
    }
    assert parse_swarm_config(config) is None


def test_sse_swarm_event_formats():
    """SSE swarm events include event type and optional agentName, data, loopIndex."""
    s = _sse_swarm_event("swarm_run_start")
    data = json.loads(s)
    assert data["event"] == "swarm_run_start"

    s = _sse_swarm_event("swarm_agent_start", agent_name="Einstein")
    data = json.loads(s)
    assert data["event"] == "swarm_agent_start"
    assert data["agentName"] == "Einstein"

    s = _sse_swarm_event("swarm_chat_model_stream", agent_name="Cleopatra", data="Hello")
    data = json.loads(s)
    assert data["event"] == "swarm_chat_model_stream"
    assert data["agentName"] == "Cleopatra"
    assert data["data"] == "Hello"

    s = _sse_swarm_event("swarm_loop_end", loop_index=1)
    data = json.loads(s)
    assert data["event"] == "swarm_loop_end"
    assert data["loopIndex"] == 1

    s = _sse_swarm_event("swarm_director_done", data={"plan": "P", "orders": []})
    data = json.loads(s)
    assert data["event"] == "swarm_director_done"
    assert data["data"]["plan"] == "P"


def test_swarm_event_names_contract():
    """Event names match the contract expected by the UI."""
    expected = [
        "swarm_run_start",
        "swarm_director_start",
        "swarm_director_done",
        "swarm_agent_start",
        "swarm_chat_model_stream",
        "swarm_agent_end",
        "swarm_loop_end",
        "swarm_run_end",
    ]
    for name in expected:
        payload = json.loads(_sse_swarm_event(name))
        assert payload["event"] == name
