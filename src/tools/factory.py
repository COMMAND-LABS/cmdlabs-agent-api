"""
Tool Factory

Creates tool instances from agent config definitions.
Supports v1 (knowledgeBases) and v2+ (tools) config formats.
"""
from typing import Dict, Any, List, Optional
from langchain_core.tools import StructuredTool
from .registry import ToolRegistry
from .agent_config_versions import extract_tool_configs_for_agent


async def create_tool_from_config(
    tool_config: Dict[str, Any],
    account_id: int,
    db: Any,
    auth_token: Optional[str] = None,
    **kwargs
) -> Optional[StructuredTool]:
    """
    Create a tool from a v2 tool configuration.
    
    Args:
        tool_config: Tool configuration dict with 'type' field
        account_id: Account ID for fetching credentials
        db: Database session
        auth_token: Authentication token (JWT or API key)
        **kwargs: Additional context passed to tool builder
        
    Returns:
        StructuredTool instance or None if type not supported
        
    Example:
        tool_config = {
            "type": "vectorSearch",
            "provider": "pinecone",
            "index": "my-index",
            "namespace": "docs",
            "topK": 10
        }
        tool = await create_tool_from_config(tool_config, account_id, db, auth_token)
    """
    tool_type = tool_config.get('type')
    
    if not tool_type:
        print(f"[TOOL FACTORY] Error: Tool config missing 'type' field: {tool_config}")
        return None
    
    # Get the builder for this tool type
    builder = ToolRegistry.get_builder(tool_type)
    
    if not builder:
        print(f"[TOOL FACTORY] Warning: Unknown tool type '{tool_type}'. Registered types: {ToolRegistry.list_types()}")
        return None
    
    # Call the builder
    try:
        print(f"[TOOL FACTORY] Creating tool of type: {tool_type}")
        tool = await builder(
            tool_config=tool_config,
            account_id=account_id,
            db=db,
            auth_token=auth_token,
            **kwargs
        )
        return tool
    except Exception as e:
        print(f"[TOOL FACTORY] Error creating tool of type '{tool_type}': {e}")
        import traceback
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
    Create all tools from an agent configuration.
    
    Supports v1 (knowledgeBases) and v2+ (tools) config formats.
    
    Args:
        agent_config: Full agent config with 'version' and 'data'
        account_id: Account ID for fetching credentials
        db: Database session
        auth_token: Authentication token (JWT or API key)
        **kwargs: Additional context passed to tool builders
        
    Returns:
        List of StructuredTool instances
        
    Example:
        # v2 config
        agent_config = {
            "schema": "agent_config",
            "version": 2,
            "data": {
                "systemPrompt": "...",
                "tools": [
                    {"type": "vectorSearch", "provider": "pinecone", ...},
                    {"type": "webSearch", "provider": "serper", ...}
                ]
            }
        }
        
        # v1 config (backwards compatible)
        agent_config = {
            "schema": "agent_config",
            "version": 1,
            "data": {
                "systemPrompt": "...",
                "knowledgeBases": [
                    {"provider": "pinecone", "index": "...", ...}
                ]
            }
        }
        
        tools = await create_tools_from_agent_config(agent_config, account_id, db, auth_token)
    """
    tools = []
    version, version_label, tool_configs = extract_tool_configs_for_agent(agent_config)
    
    print(f"[TOOL FACTORY] Creating tools from agent config v{version}")
    print(f"[TOOL FACTORY] Received kwargs: {list(kwargs.keys())}")
    print(f"[TOOL FACTORY] Found {len(tool_configs)} tools ({version_label} format)")

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

    print(f"[TOOL FACTORY] Created {len(tools)} tools successfully")
    return tools
