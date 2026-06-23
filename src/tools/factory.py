"""
Tool Factory

Dynamically creates LangChain StructuredTool instances from a v4 agent config.
"""
import logging
from typing import Any

from langchain_core.tools import StructuredTool

from .registry import ToolRegistry

logger = logging.getLogger(__name__)


def _extract_tool_configs(agent_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the list of raw tool config dicts from a v4 agent config."""
    tools = agent_config.get("data", {}).get("tools", [])
    return [t for t in tools if isinstance(t, dict)] if isinstance(tools, list) else []


async def _build_tool(
    tool_config: dict[str, Any],
    account_id: int,
    db: Any,
    auth_token: str | None = None,
    **kwargs
) -> StructuredTool | None:
    tool_type = tool_config.get("type")

    if not tool_type:
        logger.error(f"[TOOL FACTORY] Error: tool config missing 'type' field: {tool_config}")
        return None

    builder = ToolRegistry.get_builder(tool_type)
    if not builder:
        logger.warning(f"[TOOL FACTORY] Warning: unknown tool type '{tool_type}'. "
                       f"Registered: {ToolRegistry.list_types()}")
        return None

    try:
        return await builder(
            tool_config=tool_config,
            account_id=account_id,
            db=db,
            auth_token=auth_token,
            **kwargs
        )
    except Exception as e:
        logger.exception(f"[TOOL FACTORY] Error building tool '{tool_type}': {e}")
        return None


async def create_tools_from_agent_config(
    agent_config: dict[str, Any],
    account_id: int,
    db: Any,
    auth_token: str | None = None,
    **kwargs
) -> list[StructuredTool]:
    """
    Build all LangChain tools declared in a v4 agent config.

    Misconfigured or unknown tool entries are skipped with a warning rather
    than raising, so a single bad tool never kills the whole agent.
    """
    tool_configs = _extract_tool_configs(agent_config)

    tools = []
    for cfg in tool_configs:
        tool = await _build_tool(
            tool_config=cfg,
            account_id=account_id,
            db=db,
            auth_token=auth_token,
            **kwargs
        )
        if tool:
            tools.append(tool)

    logger.info(f"[TOOL FACTORY] {len(tools)}/{len(tool_configs)} tool(s) built.")
    return tools
