from __future__ import annotations

from typing import Any


def extract_tool_configs(config_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Map v1 knowledge bases to tool configs."""
    knowledge_bases = config_data.get("knowledgeBases", [])
    tool_configs: list[dict[str, Any]] = []

    for kb in knowledge_bases:
        namespace = kb.get("namespace")
        tool_configs.append(
            {
                "type": "vectorSearch",
                "provider": kb.get("provider"),
                "index": kb.get("index"),
                "namespace": namespace,
                "description": kb.get("description", f"Search the {namespace} knowledge base"),
                "topK": 10,
            }
        )

    return tool_configs
