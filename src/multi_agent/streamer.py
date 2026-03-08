"""
Stream tokens from a single agent's LLM.

No graph, no event attribution — just direct ``llm.astream()`` and yield
each token as it arrives.
"""

import re
from typing import AsyncGenerator, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

_AGENT_PREFIX_RE = re.compile(r"^\[(.+?)\]:\s*")


def _build_messages(
    system_prompt: str,
    history: List[Dict[str, str]],
    prompt: str,
    agent_name: Optional[str] = None,
) -> list:
    """Convert history dicts + prompt into LangChain message objects.

    When *agent_name* is supplied, assistant messages from *this* agent
    become ``AIMessage`` (the LLM's own past output) and messages from
    *other* agents become ``HumanMessage`` so the LLM sees them as
    something someone else said.
    """
    messages = [SystemMessage(content=system_prompt)]
    for h in history:
        role = h.get("role", "user")
        content = h.get("content", "")

        if role == "user":
            messages.append(HumanMessage(content=content))
            continue

        if agent_name:
            m = _AGENT_PREFIX_RE.match(content)
            if m and m.group(1) == agent_name:
                messages.append(AIMessage(content=content[m.end():]))
            else:
                messages.append(HumanMessage(content=content))
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
    async for chunk in llm.astream(messages):
        token = chunk.content if isinstance(chunk.content, str) else ""
        if token:
            yield token
