from __future__ import annotations

from typing import Any

from . import v1, v2_plus


def extract_tool_configs_for_agent(agent_config: dict[str, Any]) -> tuple[int, str, list[dict[str, Any]]]:
    """
    Return (version, label, tool_configs) for an agent config.

    Labels are intended for diagnostics/logging only.
    """
    version = agent_config.get("version", 1)
    config_data = agent_config.get("data", {})
    if not isinstance(config_data, dict):
        config_data = {}

    if version == 1:
        return version, "v1", v1.extract_tool_configs(config_data)

    if version >= 2:
        return version, "v2+", v2_plus.extract_tool_configs(config_data)

    return version, "unsupported", []
