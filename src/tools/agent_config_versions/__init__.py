from __future__ import annotations

from typing import Any

from . import v2_plus


def extract_tool_configs_for_agent(agent_config: dict[str, Any]) -> tuple[int, str, list[dict[str, Any]]]:
    """Return (version, label, tool_configs) for a v4 agent config."""
    version = agent_config.get("version", 4)
    config_data = agent_config.get("data", {})
    if not isinstance(config_data, dict):
        config_data = {}
    return version, "v4", v2_plus.extract_tool_configs(config_data)
