"""
Route a user message to the appropriate agent(s) in a group chat.

Makes a single fast LLM call with JSON structured output to decide
who should speak next. Returns an ordered list of display names.
"""

import json
from typing import List, Dict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage


async def route_message(
    prompt: str,
    history: List[Dict[str, str]],
    agents: List[Dict[str, str]],
    llm: BaseChatModel,
) -> List[str]:
    """Return an ordered list of agent display names that should respond.

    *agents* is a list of dicts with keys ``name`` and ``description``.
    Usually returns a single name; multiple only when the user explicitly
    addresses more than one participant.
    """
    agent_list = "\n".join(
        f"- {a['name']}: {a.get('description') or 'no description'}"
        for a in agents
    )
    names_json = json.dumps([a["name"] for a in agents])

    history_block = ""
    if history:
        lines = []
        for h in history[-20:]:
            role = h.get("role", "user")
            speaker = h.get("agent_name")
            label = speaker if speaker else role
            lines.append(f"{label}: {h.get('content', '')}")
        history_block = "\n".join(lines) + "\n\n"

    system = (
        "You decide who speaks next in a group chat.\n\n"
        f"Participants:\n{agent_list}\n\n"
        "Rules:\n"
        "- Return ONLY a JSON array of participant names in speaking order.\n"
        "- Pick whoever the human is talking to or about.\n"
        "- If the message is vague (e.g. \"hey\"), pick one natural responder.\n"
        "- Include multiple names if the human addresses multiple people.\n"
        "- If the human asks for a back-and-forth conversation (e.g. \"talk for 3 turns each\"), "
        "repeat names in alternating order for the requested number of turns, "
        "e.g. [\"A\",\"B\",\"A\",\"B\",\"A\",\"B\"].\n"
        f"- Valid names: {names_json}\n"
        "- No explanation, just the JSON array."
    )

    user_content = f"{history_block}Human: {prompt}"

    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=user_content),
    ])

    text = response.content.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            valid_names = {a["name"] for a in agents}
            result = [n for n in parsed if n in valid_names]
            if result:
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: return the first agent
    return [agents[0]["name"]]
