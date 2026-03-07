"""
In-house hierarchical swarm runner: director produces plan + orders, workers execute with streaming.
Uses LangChain LLMs; no external swarms package. Pushes events to a queue for SSE.
"""
import json
import queue
import re
from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .config import SwarmConfig


# Sentinel for queue end
SWARM_END = "_swarm_end"


def _parse_director_output(raw: str) -> tuple[str, List[Dict[str, str]]]:
    """Parse director response into plan and orders. Expects JSON with plan and orders keys."""
    plan = ""
    orders: List[Dict[str, str]] = []
    try:
        # Try to find JSON block
        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        data = json.loads(stripped)
        plan = data.get("plan") or ""
        raw_orders = data.get("orders") or []
        for o in raw_orders:
            if isinstance(o, dict) and o.get("agent_name") and o.get("task"):
                orders.append({"agent_name": str(o["agent_name"]), "task": str(o["task"])})
    except (json.JSONDecodeError, TypeError):
        # Fallback: treat whole response as plan, assign to first worker if we have one
        plan = raw
    return plan, orders


def _run_director(
    director_llm: BaseChatModel,
    director_name: str,
    worker_names: List[str],
    history: str,
    task: str,
    director_system_prompt: Optional[str],
) -> tuple[str, List[Dict[str, str]]]:
    """Run director once to get plan and orders. Returns (plan, orders)."""
    system = director_system_prompt or (
        "You are the Director. Given the conversation history and the user's latest message, "
        "output a JSON object with two keys: \"plan\" (string, your brief plan) and \"orders\" "
        "(list of objects, each with \"agent_name\" and \"task\"). "
        "Assign each order to exactly one of the available agents. "
        "Available agents: " + ", ".join(worker_names)
    )
    user_content = f"Conversation history:\n{history}\n\nUser message: {task}\n\nOutput JSON only."
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_content),
    ]
    response = director_llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)
    return _parse_director_output(raw)


def _stream_worker_sync(
    worker_llm: BaseChatModel,
    agent_name: str,
    task: str,
    history: str,
    streaming_callback: Callable[[str, str, bool], None],
    system_prompt: Optional[str] = None,
) -> str:
    """Run worker with streaming; call callback(agent_name, chunk, is_final). Returns full output."""
    system = system_prompt or f"You are {agent_name}. Respond to the given task based on the conversation context."
    content = f"Conversation history:\n{history}\n\nYour task: {task}"
    messages = [SystemMessage(content=system), HumanMessage(content=content)]
    full = ""
    if hasattr(worker_llm, "stream"):
        for chunk in worker_llm.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                full += chunk.content
                streaming_callback(agent_name, chunk.content, False)
        streaming_callback(agent_name, "", True)
    else:
        response = worker_llm.invoke(messages)
        full = response.content if hasattr(response, "content") else str(response)
        if full:
            streaming_callback(agent_name, full, False)
        streaming_callback(agent_name, "", True)
    return full


def run_swarm_streaming(
    task: str,
    history: str,
    swarm_config: SwarmConfig,
    director_llm: BaseChatModel,
    worker_llms: Dict[str, BaseChatModel],
    event_queue: queue.Queue,
    max_loops: Optional[int] = None,
) -> Any:
    """
    Run one or more loops: director -> orders -> workers (streaming). Pushes events to event_queue.
    Events pushed: ("swarm_run_start",), ("swarm_director_start",), ("swarm_director_done", payload),
    ("stream", agent_name, chunk, is_final)*, ("swarm_loop_end", loop_index), ("swarm_run_end", result), (SWARM_END,).
    """
    loops = max_loops if max_loops is not None else swarm_config.max_loops
    worker_names = [w.agent_name for w in swarm_config.workers]
    last_output: Any = None

    try:
        event_queue.put(("swarm_run_start",))
        event_queue.put(("swarm_director_start",))

        for loop_index in range(loops):
            loop_task = task if loop_index == 0 else (
                f"Previous result: {last_output}\n\nOriginal request: {task}\n\nContinue or refine."
            )
            plan, orders = _run_director(
                director_llm,
                swarm_config.director.name,
                worker_names,
                history,
                loop_task,
                swarm_config.director.system_prompt,
            )
            event_queue.put((
                "swarm_director_done",
                {"plan": plan, "orders": [{"agent_name": o["agent_name"], "task": o["task"]} for o in orders]},
            ))

            def streaming_callback(agent_name: str, chunk: str, is_final: bool) -> None:
                event_queue.put(("stream", agent_name, chunk, is_final))

            worker_specs = {w.agent_name: w for w in swarm_config.workers}
            outputs: List[str] = []
            for order in orders:
                agent_name = order["agent_name"]
                worker_task = order["task"]
                llm = worker_llms.get(agent_name)
                if not llm:
                    outputs.append(f"[{agent_name} not configured]")
                    continue
                spec = worker_specs.get(agent_name)
                sys_prompt = spec.system_prompt if spec else None
                full = _stream_worker_sync(llm, agent_name, worker_task, history, streaming_callback, system_prompt=sys_prompt)
                outputs.append(full)

            last_output = "\n\n".join(f"{orders[i]['agent_name']}: {outputs[i]}" for i in range(len(outputs)))
            event_queue.put(("swarm_loop_end", loop_index + 1))

        event_queue.put(("swarm_run_end", last_output))
    except Exception as e:
        event_queue.put(("error", str(e)))
    finally:
        event_queue.put((SWARM_END, None))
