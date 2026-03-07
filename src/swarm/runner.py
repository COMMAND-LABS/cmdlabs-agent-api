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


SWARM_END = "_swarm_end"

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(raw: str) -> Optional[dict]:
    """Extract a JSON object from LLM text that may contain markdown fences or prose."""
    stripped = raw.strip()

    # Fast path: entire response is valid JSON
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try fenced code block first (```json ... ```)
    m = _JSON_BLOCK_RE.search(stripped)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            pass

    # Last resort: grab the first { … } span
    m = _JSON_OBJECT_RE.search(stripped)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def _parse_director_output(raw: str, worker_names: List[str]) -> tuple[str, List[Dict[str, str]]]:
    """Parse director response into plan and orders. Expects JSON with plan and orders keys."""
    plan = ""
    orders: List[Dict[str, str]] = []

    data = _extract_json(raw)
    if data and isinstance(data, dict):
        plan = data.get("plan") or ""
        raw_orders = data.get("orders") or []
        for o in raw_orders:
            if not isinstance(o, dict):
                continue
            name = o.get("agent_name") or o.get("agentName") or o.get("agent") or ""
            task = o.get("task") or o.get("instruction") or ""
            if name and task:
                orders.append({"agent_name": str(name), "task": str(task)})
    else:
        # JSON extraction failed — use raw text as plan and assign to first worker
        plan = raw
        if worker_names:
            orders.append({"agent_name": worker_names[0], "task": raw})
            print(f"[SWARM DIRECTOR] JSON parse failed; assigned full response to {worker_names[0]}")

    return plan, orders


def _extract_content(response) -> str:
    """Safely extract text from an LLM response (handles str, list-of-blocks, etc.)."""
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
        return "".join(parts)
    return str(content) if content else ""


def _run_director(
    director_llm: BaseChatModel,
    director_name: str,
    worker_names: List[str],
    history: str,
    task: str,
    director_system_prompt: Optional[str],
) -> tuple[str, List[Dict[str, str]]]:
    """Run director once to get plan and orders. Returns (plan, orders)."""
    agents_list = ", ".join(f'"{n}"' for n in worker_names)
    system = director_system_prompt or (
        "You are the Director of a multi-agent swarm. "
        "Given the conversation history and the user's latest message, "
        "you MUST respond with ONLY a JSON object (no markdown, no explanation) "
        "with exactly two keys:\n"
        '  "plan": a brief string describing your plan,\n'
        '  "orders": an array of objects, each with "agent_name" (string) and "task" (string).\n'
        f"Available agents: [{agents_list}]\n"
        "You MUST assign at least one order to one of the available agents.\n"
        "Example:\n"
        '{"plan":"Summarize the topic","orders":[{"agent_name":"Researcher","task":"Find key facts"}]}'
    )
    user_content = f"Conversation history:\n{history}\n\nUser message: {task}"
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_content),
    ]
    response = director_llm.invoke(messages)
    raw = _extract_content(response)
    print(f"[SWARM DIRECTOR] Raw response ({len(raw)} chars): {raw[:500]}")
    return _parse_director_output(raw, worker_names)


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
            text = _extract_content(chunk)
            if text:
                full += text
                streaming_callback(agent_name, text, False)
        streaming_callback(agent_name, "", True)
    else:
        response = worker_llm.invoke(messages)
        full = _extract_content(response)
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

        for loop_index in range(loops):
            event_queue.put(("swarm_director_start",))

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
            print(f"[SWARM] Loop {loop_index + 1}: plan={plan[:100]!r}, orders={len(orders)}")
            event_queue.put((
                "swarm_director_done",
                {"plan": plan, "orders": [{"agent_name": o["agent_name"], "task": o["task"]} for o in orders]},
            ))

            if not orders:
                print(f"[SWARM] Loop {loop_index + 1}: No orders from director, skipping workers")
                event_queue.put(("swarm_loop_end", loop_index + 1))
                continue

            def streaming_callback(agent_name: str, chunk: str, is_final: bool) -> None:
                event_queue.put(("stream", agent_name, chunk, is_final))

            worker_specs = {w.agent_name: w for w in swarm_config.workers}
            outputs: List[str] = []
            for order in orders:
                agent_name = order["agent_name"]
                worker_task = order["task"]
                llm = worker_llms.get(agent_name)
                if not llm:
                    print(f"[SWARM] Worker {agent_name!r} not found in worker_llms keys: {list(worker_llms.keys())}")
                    outputs.append(f"[{agent_name} not configured]")
                    continue
                spec = worker_specs.get(agent_name)
                sys_prompt = spec.system_prompt if spec else None
                full = _stream_worker_sync(llm, agent_name, worker_task, history, streaming_callback, system_prompt=sys_prompt)
                outputs.append(full)

            last_output = "\n\n".join(f"{orders[i]['agent_name']}: {outputs[i]}" for i in range(len(outputs)))
            event_queue.put(("swarm_loop_end", loop_index + 1))

        event_queue.put(("swarm_run_end", last_output or ""))
    except Exception as e:
        import traceback
        traceback.print_exc()
        event_queue.put(("error", str(e)))
    finally:
        event_queue.put((SWARM_END, None))
