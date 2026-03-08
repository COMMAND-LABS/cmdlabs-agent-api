"""
Stream tokens from a single agent's LLM.

No graph, no event attribution — just direct ``llm.astream()`` and yield
each token as it arrives.
"""

from typing import AsyncGenerator, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


def _build_messages(
    system_prompt: str,
    history: List[Dict[str, str]],
    prompt: str,
) -> list:
    """Convert history dicts + prompt into LangChain message objects."""
    messages = [SystemMessage(content=system_prompt)]
    for h in history:
        role = h.get("role", "user")
        content = h.get("content", "")
        if role == "user":
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
) -> AsyncGenerator[str, None]:
    """Yield content tokens from the agent's LLM one at a time."""
    messages = _build_messages(system_prompt, history, prompt)
    async for chunk in llm.astream(messages):
        token = chunk.content if isinstance(chunk.content, str) else ""
        if token:
            yield token
