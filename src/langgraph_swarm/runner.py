"""
LangGraph supervisor swarm – token-level streaming via astream_events.

Worker agents are subgraphs whose internal LLM node is always called "agent".
We track which worker is active via a nesting-depth counter on on_chain_start/end,
then attribute on_chat_model_stream tokens to the active worker. A background task
+ asyncio.Queue keeps the SSE connection alive with keepalive heartbeats.
"""

import asyncio
import re
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

_SENTINEL = object()


def _to_node_name(display_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", display_name).strip("_").lower()
    return slug or "agent"


def _build_graph(
    supervisor_llm: BaseChatModel,
    supervisor_prompt: str,
    worker_configs: List[Dict[str, Any]],
    worker_llms: Dict[str, BaseChatModel],
    output_mode: str = "last_message",
):
    agents = []
    for wc in worker_configs:
        name = wc["node_name"]
        agents.append(
            create_react_agent(
                model=worker_llms[name],
                tools=[],
                name=name,
                prompt=wc.get("system_prompt") or f"You are {wc.get('display_name', name)}.",
            )
        )
    return create_supervisor(
        agents,
        model=supervisor_llm,
        prompt=supervisor_prompt,
        output_mode=output_mode,
        add_handoff_back_messages=False,
    ).compile()


def _event_node(ev: dict) -> str:
    """Node/graph name that produced this event. Prefer name, then metadata.langgraph_node."""
    name = ev.get("name") or ""
    if name:
        return name
    return (ev.get("metadata") or {}).get("langgraph_node", "")


async def _stream_tokens(app, messages, worker_names, node_to_display, q: asyncio.Queue):
    """Run astream_events in a task; push swarm SSE dicts into *q*."""
    worker_depth: Dict[str, int] = {}
    active_worker: Optional[str] = None
    current_streaming: Optional[str] = None
    try:
        print(f"[LANGGRAPH] astream_events starting, workers={worker_names}", flush=True)
        token_count = 0
        async for ev in app.astream_events(
            {"messages": messages},
            version="v2",
            config={"recursion_limit": 30},
        ):
            kind = ev.get("event", "")
            node = _event_node(ev)

            if kind == "on_chain_start" and node in worker_names:
                worker_depth[node] = worker_depth.get(node, 0) + 1
                if active_worker != node:
                    print(f"[LANGGRAPH] worker active: {node} depth={worker_depth[node]}", flush=True)
                    active_worker = node

            if kind == "on_chain_end" and node in worker_names:
                worker_depth[node] = max(0, worker_depth.get(node, 0) - 1)
                if worker_depth[node] == 0 and active_worker == node:
                    print(f"[LANGGRAPH] worker done: {node} tokens_so_far={token_count}", flush=True)
                    active_worker = None

            # Token from worker: accept on_chat_model_stream when we're inside a worker.
            # Node may be "agent" (react agent inner node) or another runnable name.
            if kind == "on_chat_model_stream" and active_worker:
                raw = ev.get("data")
                if isinstance(raw, dict) and "chunk" in raw:
                    chunk = raw["chunk"]
                else:
                    chunk = raw
                if not isinstance(chunk, AIMessageChunk):
                    continue
                token = chunk.content if isinstance(chunk.content, str) else ""
                if not token:
                    continue

                display = node_to_display.get(active_worker, active_worker)

                if current_streaming != active_worker:
                    if current_streaming is not None:
                        await q.put({"event": "swarm_agent_end",
                                     "agentName": node_to_display.get(current_streaming, current_streaming)})
                    current_streaming = active_worker
                    await q.put({"event": "swarm_agent_start", "agentName": display})

                await q.put({"event": "swarm_chat_model_stream", "agentName": display, "data": token})
                token_count += 1

        if current_streaming is not None:
            await q.put({"event": "swarm_agent_end",
                         "agentName": node_to_display.get(current_streaming, current_streaming)})

        print(f"[LANGGRAPH] astream_events done – {token_count} tokens streamed", flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[LANGGRAPH] stream error: {exc}", flush=True)
        await q.put({"event": "error", "data": {"error": "Stream error", "message": str(exc)}})
    finally:
        await q.put(_SENTINEL)


async def stream_langgraph_swarm(
    *,
    prompt: str,
    history: List[Dict[str, str]],
    supervisor_llm: BaseChatModel,
    supervisor_prompt: str,
    worker_configs: List[Dict[str, Any]],
    worker_llms: Dict[str, BaseChatModel],
    output_mode: str = "last_message",
) -> AsyncGenerator[Dict[str, Any], None]:
    node_to_display = {
        wc["node_name"]: wc.get("display_name", wc["node_name"])
        for wc in worker_configs
    }
    worker_names = set(node_to_display.keys())

    app = _build_graph(supervisor_llm, supervisor_prompt, worker_configs,
                       worker_llms, output_mode)

    messages = [{"role": h.get("role", "user"), "content": h.get("content", "")}
                for h in history]
    messages.append({"role": "user", "content": prompt})

    yield {"event": "swarm_run_start"}

    q: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(
        _stream_tokens(app, messages, worker_names, node_to_display, q))

    agent_outputs: Dict[str, str] = {}

    while True:
        try:
            evt = await asyncio.wait_for(q.get(), timeout=2.0)
        except asyncio.TimeoutError:
            yield {"event": "swarm_keepalive"}
            continue

        if evt is _SENTINEL:
            break

        if evt.get("event") == "swarm_chat_model_stream":
            name = evt.get("agentName", "")
            agent_outputs[name] = agent_outputs.get(name, "") + evt.get("data", "")

        yield evt

    await task
    yield {"event": "swarm_run_end", "data": agent_outputs}
