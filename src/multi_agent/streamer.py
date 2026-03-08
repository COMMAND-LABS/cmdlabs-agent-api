"""
Stream tokens from a single agent's LLM.

The agent sees the full shared conversation history (system prompt +
all human/agent messages) and responds to whatever the last message is.
No synthetic prompts are injected.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, AsyncGenerator, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

if TYPE_CHECKING:
    from .session_logger import SessionLogger

_INVALID_NAME_CHARS = re.compile(r"[\s<|\\/>]+")


def _build_messages(
    system_prompt: str,
    history: List[Dict[str, str]],
    agent_name: Optional[str] = None,
) -> list:
    """Convert the shared history into LangChain message objects.

    Messages from *other* agents get the ``name`` field set so the LLM
    can tell who said what.  The current agent's own past messages stay
    as plain ``AIMessage`` to avoid smaller models echoing the name.
    """
    messages: list = [SystemMessage(content=system_prompt)]
    for h in history:
        role = h.get("role", "user")
        content = h.get("content", "")

        if role == "user":
            messages.append(HumanMessage(content=content))
            continue

        speaker = h.get("agent_name")
        if speaker and speaker != agent_name:
            safe_name = _INVALID_NAME_CHARS.sub("_", speaker)
            messages.append(AIMessage(content=content, name=safe_name))
        else:
            messages.append(AIMessage(content=content))

    return messages


async def stream_agent(
    system_prompt: str,
    history: List[Dict[str, str]],
    llm: BaseChatModel,
    agent_name: Optional[str] = None,
    session_logger: SessionLogger | None = None,
) -> AsyncGenerator[str, None]:
    """Yield content tokens from the agent's LLM one at a time."""
    messages = _build_messages(system_prompt, history, agent_name=agent_name)
    if session_logger:
        session_logger.log_llm_messages(f"agent:{agent_name or '?'}", messages)
    async for chunk in llm.astream(messages):
        token = chunk.content if isinstance(chunk.content, str) else ""
        if token:
            yield token
