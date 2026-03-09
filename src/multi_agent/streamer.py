"""
Stream tokens from a single agent's LLM.

The agent sees the full shared conversation history (system prompt +
all human/agent messages) and responds to whatever the last message is.
No synthetic prompts are injected.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, AsyncGenerator, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.routers.swarms.policy import build_agent_reply_request
from src.routers.swarms.types import AgentDefinition, ConversationEntry

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


async def stream_agent(
    system_prompt: str,
    history: List[Dict[str, str]],
    llm: BaseChatModel,
    agent_name: Optional[str] = None,
    session_logger: SessionLogger | None = None,
) -> AsyncGenerator[str, None]:
    """Yield content tokens from the agent's LLM one at a time."""
    request = build_agent_reply_request(
        agent=AgentDefinition(
            name=agent_name or "?",
            description="",
            model="",
            system_prompt=system_prompt,
        ),
        history=_to_conversation_entries(history),
    )
    messages = _to_langchain_messages(request.messages)
    if session_logger:
        session_logger.log_llm_messages(request.label, messages)
    async for chunk in llm.astream(messages):
        token = chunk.content if isinstance(chunk.content, str) else ""
        if token:
            yield token
