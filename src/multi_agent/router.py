"""
Route a user message to the appropriate agent in a group chat.

A single fast LLM call examines the full conversation history and
decides who (if anyone) should speak next.  Returns one name or
an empty list.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

if TYPE_CHECKING:
    from .session_logger import SessionLogger


def _format_history(history: List[Dict[str, str]]) -> str:
    lines = []
    for h in history[-20:]:
        speaker = h.get("agent_name")
        if speaker:
            label = f"[ai: {speaker}]"
        else:
            label = "[human]"
        lines.append(f"{label} {h.get('content', '')}")
    return "\n".join(lines)


async def route_next_speaker(
    history: List[Dict[str, str]],
    agents: List[Dict[str, str]],
    llm: BaseChatModel,
    session_logger: SessionLogger | None = None,
) -> List[str]:
    """Return the single agent that should speak next, or ``[]`` to stop.

    Works for both the initial turn (human just spoke) and continuation
    turns (an agent just spoke and someone may need to reply).
    """
    agent_list = "\n".join(
        f"- {a['name']}: {a.get('description') or 'no description'}"
        for a in agents
    )
    names_json = json.dumps([a["name"] for a in agents])

    system = (
        "You decide who should speak next in a group chat.\n\n"
        f"Participants:\n{agent_list}\n\n"
        "Look at the conversation history and decide which participant "
        "(if any) should respond to the LAST message.\n\n"
        "Rules:\n"
        "- If the last message is from a human, pick whoever they are talking to.\n"
        "- If the last message is from a participant and it asks a question to "
        "or addresses another participant, pick that participant.\n"
        "- If the conversation has reached a natural pause, return [].\n"
        "- NEVER pick the same participant who sent the last message.\n"
        "- Return ONLY a JSON array with at most ONE name, or [].\n"
        f"- Valid names: {names_json}\n"
        "- No explanation, just the JSON array."
    )

    history_block = _format_history(history)

    messages = [SystemMessage(content=system), HumanMessage(content=history_block)]
    if session_logger:
        session_logger.log_llm_messages("router", messages)

    response = await llm.ainvoke(messages)

    text = response.content.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            valid_names = {a["name"] for a in agents}
            result = [n for n in parsed if n in valid_names]
            if result:
                return result[:1]
    except (json.JSONDecodeError, TypeError):
        pass

    return []
