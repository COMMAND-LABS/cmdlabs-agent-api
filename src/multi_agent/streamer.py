"""
Stream tokens from a single agent's LLM.

No graph, no event attribution — just direct ``llm.astream()`` and yield
each token as it arrives.
"""

import re
from typing import AsyncGenerator, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

_INVALID_NAME_CHARS = re.compile(r"[\s<|\\/>]+")


def _build_messages(
    system_prompt: str,
    history: List[Dict[str, str]],
    prompt: str,
    agent_name: Optional[str] = None,
) -> list:
    """Convert history dicts + prompt into LangChain message objects.

    Each history dict has ``role``, ``content``, and an optional
    ``agent_name``.  Messages from *other* agents get the ``name``
    field set so the LLM can tell who said what.  The current agent's
    own past messages stay as plain ``AIMessage`` — the model already
    treats those as its own prior output, and adding ``name`` to them
    confuses smaller models into echoing the name as content.
    """
    messages = [SystemMessage(content=system_prompt)]
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

    messages.append(HumanMessage(content=prompt))
    return messages


async def stream_agent(
    system_prompt: str,
    history: List[Dict[str, str]],
    prompt: str,
    llm: BaseChatModel,
    agent_name: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Yield content tokens from the agent's LLM one at a time."""
    messages = _build_messages(system_prompt, history, prompt, agent_name=agent_name)
    config: Dict = {}
    if agent_name:
        config["run_name"] = agent_name
        config["metadata"] = {"agent_name": agent_name}
    async for chunk in llm.astream(messages, config=config or None):
        token = chunk.content if isinstance(chunk.content, str) else ""
        if token:
            yield token
