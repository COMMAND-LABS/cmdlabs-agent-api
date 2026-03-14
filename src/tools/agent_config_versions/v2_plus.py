from __future__ import annotations

from typing import Any


def extract_tool_configs(config_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Read native tool configs for v2+ agent configs."""
    tool_configs = config_data.get("tools", [])
    if not isinstance(tool_configs, list):
        return []
    return [tool for tool in tool_configs if isinstance(tool, dict)]
