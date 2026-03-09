"""Minimal provider clients for the TTS next-turn swarm endpoint."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

if False:  # pragma: no cover
    from src.multi_agent.session_logger import SessionLogger


_INVALID_NAME_CHARS = re.compile(r"[\s<|\\/>]+")


def _sanitize_name(name: str) -> str:
    return _INVALID_NAME_CHARS.sub("_", name)


def _openai_token_limit_field(model: str) -> str:
    normalized = model.strip().lower()
    if normalized.startswith("gpt-5"):
        return "max_completion_tokens"
    return "max_tokens"


def _is_gpt5_model(model: str) -> bool:
    return model.strip().lower().startswith("gpt-5")


def _openai_supports_temperature(model: str) -> bool:
    return not _is_gpt5_model(model)


def _provider_message(role: str, content: str, *, name: Optional[str] = None) -> Dict[str, str]:
    message: Dict[str, str] = {"role": role, "content": content}
    if name:
        message["name"] = _sanitize_name(name)
    return message


def _anthropic_payload(messages: List[Dict[str, str]]) -> tuple[str, List[Dict[str, str]]]:
    system_parts: list[str] = []
    chat_messages: list[Dict[str, str]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        name = message.get("name")
        if role == "system":
            system_parts.append(content)
            continue
        if name:
            content = f"[{name}] {content}"
        chat_messages.append({"role": role, "content": content})
    return "\n\n".join(system_parts), chat_messages


def _extract_anthropic_text(response) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", "") == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts).strip()


def _response_to_loggable_body(response) -> object:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    return response


def _responses_api_payload(
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
) -> Dict[str, object]:
    instructions_parts: list[str] = []
    input_messages: list[Dict[str, str]] = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        name = message.get("name")

        if role == "system":
            instructions_parts.append(content)
            continue

        if name:
            content = f"[{name}] {content}"
        input_messages.append({"role": role, "content": content})

    request_body: Dict[str, object] = {
        "model": model,
        "input": input_messages,
        "max_output_tokens": max_tokens,
        "reasoning": {"effort": "none"},
    }
    if instructions_parts:
        request_body["instructions"] = "\n\n".join(instructions_parts)
    return request_body


async def complete_text(
    *,
    provider: str,
    model: str,
    api_key: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    session_logger: "SessionLogger | None" = None,
    request_label: str = "llm",
    log_response_body: bool = False,
) -> str:
    """Return a plain text completion for the given provider."""
    if provider == "openai":
        client = AsyncOpenAI(api_key=api_key)
        if _is_gpt5_model(model):
            request_body = _responses_api_payload(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            if session_logger:
                session_logger.log_api_request_body(request_label, provider, request_body)
            response = await client.responses.create(**request_body)
            if session_logger and log_response_body:
                session_logger.log_api_response_body(
                    request_label,
                    provider,
                    _response_to_loggable_body(response),
                )
            return getattr(response, "output_text", "").strip()

        token_limit_field = _openai_token_limit_field(model)
        request_body = {
            "model": model,
            "messages": messages,
            token_limit_field: max_tokens,
        }
        if _openai_supports_temperature(model):
            request_body["temperature"] = temperature
        if session_logger:
            session_logger.log_api_request_body(request_label, provider, request_body)
        response = await client.chat.completions.create(**request_body)
        if session_logger and log_response_body:
            session_logger.log_api_response_body(
                request_label,
                provider,
                _response_to_loggable_body(response),
            )
        return (response.choices[0].message.content or "").strip()

    if provider == "anthropic":
        system, chat_messages = _anthropic_payload(messages)
        client = AsyncAnthropic(api_key=api_key)
        request_body = {
            "model": model,
            "system": system,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if session_logger:
            session_logger.log_api_request_body(request_label, provider, request_body)
        response = await client.messages.create(**request_body)
        if session_logger and log_response_body:
            session_logger.log_api_response_body(
                request_label,
                provider,
                _response_to_loggable_body(response),
            )
        return _extract_anthropic_text(response)

    raise ValueError(f"Unsupported provider: {provider}")


def build_router_messages(history: List[Dict[str, str]], agents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    recent = history[-20:]
    agent_list = "\n".join(
        f"- {agent['name']}: {agent.get('description') or 'no description'}"
        for agent in agents
    )
    names_json = json.dumps([agent["name"] for agent in agents])
    system = (
        "You are quietly observing a small group conversation and deciding who, if anyone, "
        "would most naturally speak next.\n\n"
        f"## Participants\n\n{agent_list}\n\n"
        "Read the conversation as a natural dialogue, not as a strict turn-taking exercise.\n\n"
        "Guidance:\n"
        "- Prefer the participant who is most clearly being addressed in the latest message.\n"
        "- If the latest message names one participant, that participant should usually respond.\n"
        "- Do not force another reply just to keep the conversation going.\n"
        "- After a simple greeting or quick reply, it is often natural to stop.\n"
        "- Only choose someone else when the latest message genuinely invites or needs their response.\n"
        "- Never pick the participant who just spoke.\n"
        f"- Valid names: {names_json}\n\n"
        "Return ONLY a JSON object with this shape: "
        '{"next":["Valid Name"] or [],"reason":"short explanation"}'
    )
    messages: list[Dict[str, str]] = [_provider_message("system", system)]
    if not recent:
        messages.append(_provider_message("user", "(empty conversation)"))
        return messages

    for item in recent:
        speaker = item.get("agent_name")
        if speaker:
            messages.append(
                _provider_message(
                    "assistant",
                    item.get("content", ""),
                    name=speaker,
                )
            )
        else:
            messages.append(_provider_message("user", item.get("content", "")))
    return messages


def build_agent_messages(
    *,
    system_prompt: str,
    history: List[Dict[str, str]],
    agent_name: Optional[str],
) -> List[Dict[str, str]]:
    """Convert shared history into provider-friendly plain dict messages."""
    messages: list[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for item in history:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role == "user":
            messages.append(_provider_message("user", content))
            continue

        speaker = item.get("agent_name")
        if speaker and speaker != agent_name:
            messages.append(_provider_message("assistant", content, name=speaker))
            continue
        messages.append(_provider_message("assistant", content))
    return messages


async def route_next_speaker_plain(
    *,
    history: List[Dict[str, str]],
    agents: List[Dict[str, str]],
    provider: str,
    model: str,
    api_key: str,
    session_logger: "SessionLogger | None" = None,
) -> Tuple[List[str], str]:
    messages = build_router_messages(history, agents)
    if session_logger:
        session_logger.log_llm_messages("router", messages)

    text = await complete_text(
        provider=provider,
        model=model,
        api_key=api_key,
        messages=messages,
        temperature=0.7,
        max_tokens=128,
        session_logger=session_logger,
        request_label="router",
        log_response_body=True,
    )
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            reason = str(parsed.get("reason") or "").strip()
            next_value = parsed.get("next")
            if isinstance(next_value, list):
                valid_names = {agent["name"] for agent in agents}
                result = [name for name in next_value if name in valid_names]
                return result[:1], reason
        if isinstance(parsed, list):
            valid_names = {agent["name"] for agent in agents}
            result = [name for name in parsed if name in valid_names]
            return result[:1], ""
    except (json.JSONDecodeError, TypeError):
        pass
    return [], ""


async def generate_agent_reply_plain(
    *,
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    history: List[Dict[str, str]],
    agent_name: Optional[str],
    session_logger: "SessionLogger | None" = None,
) -> str:
    messages = build_agent_messages(
        system_prompt=system_prompt,
        history=history,
        agent_name=agent_name,
    )
    if session_logger:
        session_logger.log_llm_messages(f"agent:{agent_name or '?'}", messages)

    return await complete_text(
        provider=provider,
        model=model,
        api_key=api_key,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        session_logger=session_logger,
        request_label=f"agent:{agent_name or '?'}",
    )
