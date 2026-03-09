"""
Route a user message to the appropriate agent in a group chat.

A single fast LLM call examines the full conversation history and
decides who (if anyone) should speak next.  Returns one name or
an empty list.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.routers.swarms.policy import build_router_request, parse_router_decision
from src.routers.swarms.types import ConversationEntry

if TYPE_CHECKING:
    from .session_logger import SessionLogger


def _to_conversation_entries(history: List[Dict[str, str]]) -> list[ConversationEntry]:
    return [
        ConversationEntry(
            role=item.get("role", "user"),
            content=item.get("content", ""),
            agent_name=item.get("agent_name"),
        )
        for item in history
    ]


def _to_langchain_messages(messages: list[dict[str, str]]) -> list:
    result: list = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        name = message.get("name")
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "assistant":
            if name:
                result.append(AIMessage(content=content, name=name))
            else:
                result.append(AIMessage(content=content))
        else:
            result.append(HumanMessage(content=content))
    return result


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
    request = build_router_request(
        history=_to_conversation_entries(history),
        agents=agents,
    )
    messages = _to_langchain_messages(request.messages)
    if session_logger:
        session_logger.log_llm_messages(request.label, messages)

    response = await llm.ainvoke(messages)
    decision = parse_router_decision(response.content.strip(), (agent["name"] for agent in agents))
    return decision.next_speakers
