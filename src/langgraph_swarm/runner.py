"""
LangGraph supervisor-based swarm runner.

Uses langgraph-supervisor's create_supervisor + langgraph.prebuilt.create_react_agent
to build a hierarchical multi-agent swarm. Runs with ainvoke and returns results.
"""
import re
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor


def _to_node_name(display_name: str) -> str:
    """Convert a human-readable agent name to a valid LangGraph node / tool name."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", display_name).strip("_").lower()
    return slug or "agent"


def _extract_text(msg: BaseMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
        return "".join(parts)
    return str(content) if content else ""


def build_supervisor_graph(
    supervisor_llm: BaseChatModel,
    supervisor_prompt: Optional[str],
    worker_configs: List[Dict[str, Any]],
    worker_llms: Dict[str, BaseChatModel],
    output_mode: str = "last_message",
):
    agents = []
    for wc in worker_configs:
        node_name = wc["node_name"]
        display = wc.get("display_name", node_name)
        llm = worker_llms[node_name]
        agent = create_react_agent(
            model=llm,
            tools=[],
            name=node_name,
            prompt=wc.get("system_prompt") or f"You are {display}.",
        )
        agents.append(agent)

    workflow = create_supervisor(
        agents,
        model=supervisor_llm,
        prompt=supervisor_prompt,
        output_mode=output_mode,
        add_handoff_back_messages=False,
    )
    return workflow.compile()


async def run_langgraph_swarm(
    *,
    prompt: str,
    history: List[Dict[str, str]],
    supervisor_llm: BaseChatModel,
    supervisor_prompt: Optional[str],
    worker_configs: List[Dict[str, Any]],
    worker_llms: Dict[str, BaseChatModel],
    output_mode: str = "last_message",
) -> Dict[str, Any]:
    """
    Build and run the LangGraph supervisor graph.

    Returns dict with:
      - "messages": list of result messages
      - "agent_outputs": {display_name: text} for each worker that produced output
      - "error": error string if something went wrong
    """
    node_to_display: Dict[str, str] = {
        wc["node_name"]: wc.get("display_name", wc["node_name"])
        for wc in worker_configs
    }

    app = build_supervisor_graph(
        supervisor_llm=supervisor_llm,
        supervisor_prompt=supervisor_prompt,
        worker_configs=worker_configs,
        worker_llms=worker_llms,
        output_mode=output_mode,
    )

    messages: list = []
    for h in history:
        role = h.get("role", "user")
        content = h.get("content", "")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})

    result = await app.ainvoke({"messages": messages})

    result_messages = result.get("messages", [])
    print(f"[LANGGRAPH RUNNER] ainvoke returned {len(result_messages)} messages", flush=True)
    for i, msg in enumerate(result_messages):
        mtype = type(msg).__name__
        name = getattr(msg, "name", None)
        content = _extract_text(msg) if hasattr(msg, "content") else ""
        tc = getattr(msg, "tool_calls", None)
        print(
            f"[LANGGRAPH RUNNER]   [{i}] {mtype} name={name!r} "
            f"tool_calls={len(tc) if tc else 0} "
            f"content={content[:120]!r}",
            flush=True,
        )

    agent_outputs: Dict[str, str] = {}
    for msg in result_messages:
        if not isinstance(msg, (AIMessage, AIMessageChunk)):
            continue
        # Skip handoff tool-call messages
        if getattr(msg, "tool_calls", None):
            continue
        name = getattr(msg, "name", None) or ""
        if name == "supervisor" or not name:
            continue
        display = node_to_display.get(name, name)
        text = _extract_text(msg)
        if text:
            agent_outputs[display] = agent_outputs.get(display, "") + text

    # Fallback: if output_mode="last_message" collapsed everything and
    # no named worker messages were found, grab the last AI message that
    # isn't from the supervisor as the output.
    if not agent_outputs:
        for msg in reversed(result_messages):
            if not isinstance(msg, (AIMessage, AIMessageChunk)):
                continue
            name = getattr(msg, "name", None) or ""
            if name == "supervisor":
                continue
            text = _extract_text(msg)
            if text:
                label = node_to_display.get(name, name) if name else "agent"
                agent_outputs[label] = text
                break

    return {"messages": result_messages, "agent_outputs": agent_outputs}
