"""
Tool Factory

Dynamically creates LangChain StructuredTool instances from a v4 agent config.
"""
from typing import Dict, Any, List, Optional
from langchain_core.tools import StructuredTool
from .registry import ToolRegistry


def _extract_tool_configs(agent_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of raw tool config dicts from a v4 agent config."""
    tools = agent_config.get("data", {}).get("tools", [])
    return [t for t in tools if isinstance(t, dict)] if isinstance(tools, list) else []


async def create_tool_from_config(
    tool_config: Dict[str, Any],
    account_id: int,
    db: Any,
    auth_token: Optional[str] = None,
    **kwargs
) -> Optional[StructuredTool]:
    """
    Instantiate a single tool from its config dict.

    Looks up the builder registered for `tool_config["type"]` in ToolRegistry
    and delegates to it.  Returns None for unknown or misconfigured tool types.
    """
    tool_type = tool_config.get('type')

    if not tool_type:
        print(f"[TOOL FACTORY] Error: tool config missing 'type' field: {tool_config}")
        return None

    builder = ToolRegistry.get_builder(tool_type)

    if not builder:
        print(f"[TOOL FACTORY] Warning: unknown tool type '{tool_type}'. "
              f"Registered: {ToolRegistry.list_types()}")
        return None

    try:
        print(f"[TOOL FACTORY] Creating tool: {tool_type}")
        return await builder(
            tool_config=tool_config,
            account_id=account_id,
            db=db,
            auth_token=auth_token,
            **kwargs
        )
    except Exception as e:
        import traceback
        print(f"[TOOL FACTORY] Error creating tool '{tool_type}': {e}")
        traceback.print_exc()
        return None


async def create_tools_from_agent_config(
    agent_config: Dict[str, Any],
    account_id: int,
    db: Any,
    auth_token: Optional[str] = None,
    **kwargs
) -> List[StructuredTool]:
    """
    Build all LangChain tools declared in a v4 agent config.

    Returns only successfully constructed tools; misconfigured entries are
    skipped with a warning log rather than raising.
    """
    tool_configs = _extract_tool_configs(agent_config)
    print(f"[TOOL FACTORY] Building {len(tool_configs)} tool(s). "
          f"kwargs: {list(kwargs.keys())}")

    tools = []
    for tool_config in tool_configs:
        tool = await create_tool_from_config(
            tool_config=tool_config,
            account_id=account_id,
            db=db,
            auth_token=auth_token,
            **kwargs
        )
        if tool:
            tools.append(tool)

    print(f"[TOOL FACTORY] {len(tools)}/{len(tool_configs)} tool(s) built successfully.")
    return tools
