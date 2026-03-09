from __future__ import annotations

import json
import re
from typing import Iterable, Optional

from src.routers.swarms.langgraph_schemas import LanggraphSwarmConfigInput
from src.routers.swarms.types import AgentDefinition, ConversationEntry, LLMRequest, RouterDecision


_INVALID_NAME_CHARS = re.compile(r"[\s<|\\/>]+")
_ROUTER_MAX_MESSAGES = 20
_ROUTER_MAX_TOKENS = 256
_AGENT_MAX_TOKENS = 1024
_DEFAULT_TEMPERATURE = 0.7


def sanitize_name(name: str) -> str:
    return _INVALID_NAME_CHARS.sub("_", name)


def build_agent_definitions(swarm: LanggraphSwarmConfigInput) -> dict[str, AgentDefinition]:
    all_display_names = [worker.agentName for worker in swarm.workers]
    definitions: dict[str, AgentDefinition] = {}

    for worker in swarm.workers:
        base_prompt = worker.systemPrompt or f"You are {worker.agentName}."
        participants = ", ".join(
            f"{name} (you)" if name == worker.agentName else name
            for name in all_display_names
        )
        system_prompt = (
            f"{base_prompt}\n\n"
            f"Group conversation participants: Human, {participants}.\n"
            "- Speak naturally as yourself, in your own voice.\n"
            "- Only say your own words. Do not speak for other participants.\n"
            "- If the latest message is a simple greeting or direct check-in, a short natural reply is enough.\n"
            "- Do not force a bigger discussion unless the conversation genuinely calls for it.\n"
            "- Be concise. A few sentences is usually enough."
        )
        definitions[worker.agentName] = AgentDefinition(
            name=worker.agentName,
            description=worker.agentDescription or "",
            model=worker.modelName,
            system_prompt=system_prompt,
        )
    return definitions


def build_agent_list(agent_definitions: dict[str, AgentDefinition]) -> list[dict[str, str]]:
    return [
        {"name": agent.name, "description": agent.description}
        for agent in agent_definitions.values()
    ]


def _provider_message(role: str, content: str, *, name: Optional[str] = None) -> dict[str, str]:
    message: dict[str, str] = {"role": role, "content": content}
    if name:
        message["name"] = sanitize_name(name)
    return message


def build_router_request(
    *,
    history: list[ConversationEntry],
    agents: list[dict[str, str]],
    supervisor_prompt: Optional[str] = None,
) -> LLMRequest:
    recent = history[-_ROUTER_MAX_MESSAGES:]
    agent_list = "\n".join(
        f"- {agent['name']}: {agent.get('description') or 'no description'}"
        for agent in agents
    )
    participants_block = "- Human (user)\n" + agent_list
    names_json = json.dumps([agent["name"] for agent in agents])
    prompt_prefix = f"{supervisor_prompt.strip()}\n\n" if supervisor_prompt and supervisor_prompt.strip() else ""
    system = (
        f"{prompt_prefix}"
        "You are observing a small group conversation and deciding who, if anyone, "
        "would most naturally speak next.\n\n"
        f"## Participants\n\n{participants_block}\n\n"
        "Read the conversation as a natural dialogue, not as a strict turn-taking exercise.\n\n"
        "Guidance:\n"
        "- The ultimate goal is an aesthetically pleasing flow of conversation.\n"
        f"- Valid names: {names_json}\n\n"
        "Return ONLY a JSON object with this shape: "
        '{"next":["Valid Name"] or [],"reason":"short explanation"}'
    )

    messages: list[dict[str, str]] = [_provider_message("system", system)]
    if not recent:
        messages.append(_provider_message("user", "(empty conversation)"))
    else:
        for item in recent:
            if item.agent_name:
                messages.append(_provider_message("assistant", item.content, name=item.agent_name))
            else:
                messages.append(_provider_message("user", item.content))

    return LLMRequest(
        messages=messages,
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_ROUTER_MAX_TOKENS,
        label="router",
        log_response_body=True,
    )


def parse_router_decision(text: str, valid_names: Iterable[str]) -> RouterDecision:
    allowed_names = set(valid_names)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            reason = str(parsed.get("reason") or "").strip()
            next_value = parsed.get("next")
            if isinstance(next_value, list):
                result = [name for name in next_value if name in allowed_names]
                return RouterDecision(next_speakers=result[:1], reason=reason)
        if isinstance(parsed, list):
            result = [name for name in parsed if name in allowed_names]
            return RouterDecision(next_speakers=result[:1], reason="")
    except (json.JSONDecodeError, TypeError):
        pass
    return RouterDecision(next_speakers=[], reason="")


def build_agent_reply_request(
    *,
    agent: AgentDefinition,
    history: list[ConversationEntry],
) -> LLMRequest:
    messages: list[dict[str, str]] = [{"role": "system", "content": agent.system_prompt}]
    for item in history:
        if item.role == "user":
            messages.append(_provider_message("user", item.content))
            continue

        if item.agent_name and item.agent_name != agent.name:
            messages.append(_provider_message("assistant", item.content, name=item.agent_name))
            continue
        messages.append(_provider_message("assistant", item.content))

    return LLMRequest(
        messages=messages,
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_AGENT_MAX_TOKENS,
        label=f"agent:{agent.name}",
    )
