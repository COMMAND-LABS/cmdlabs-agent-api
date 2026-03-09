"""Provider adapters for the swarm TTS endpoints."""

from __future__ import annotations

from typing import AsyncGenerator

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from src.routers.swarms.types import LLMRequest

if False:  # pragma: no cover
    from src.multi_agent.session_logger import SessionLogger

def _openai_token_limit_field(model: str) -> str:
    normalized = model.strip().lower()
    if normalized.startswith("gpt-5"):
        return "max_completion_tokens"
    return "max_tokens"


def _is_gpt5_model(model: str) -> bool:
    return model.strip().lower().startswith("gpt-5")


def _openai_supports_temperature(model: str) -> bool:
    return not _is_gpt5_model(model)


def _anthropic_payload(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    system_parts: list[str] = []
    chat_messages: list[dict[str, str]] = []
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
    messages: list[dict[str, str]],
    max_tokens: int,
) -> dict[str, object]:
    instructions_parts: list[str] = []
    input_messages: list[dict[str, str]] = []

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

    request_body: dict[str, object] = {
        "model": model,
        "input": input_messages,
        "max_output_tokens": max_tokens,
        "reasoning": {"effort": "none"},
    }
    if instructions_parts:
        request_body["instructions"] = "\n\n".join(instructions_parts)
    return request_body


def _chat_completions_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> dict[str, object]:
    token_limit_field = _openai_token_limit_field(model)
    request_body: dict[str, object] = {
        "model": model,
        "messages": messages,
        token_limit_field: max_tokens,
    }
    if _openai_supports_temperature(model):
        request_body["temperature"] = temperature
    return request_body


async def complete_request(
    *,
    provider: str,
    model: str,
    api_key: str,
    request: LLMRequest,
    session_logger: "SessionLogger | None" = None,
) -> str:
    """Return a plain text completion for the given provider."""
    if provider == "openai":
        client = AsyncOpenAI(api_key=api_key)
        if _is_gpt5_model(model):
            request_body = _responses_api_payload(
                model=model,
                messages=request.messages,
                max_tokens=request.max_tokens,
            )
            if session_logger:
                session_logger.log_api_request_body(request.label, provider, request_body)
            response = await client.responses.create(**request_body)
            if session_logger and request.log_response_body:
                session_logger.log_api_response_body(
                    request.label,
                    provider,
                    _response_to_loggable_body(response),
                )
            return getattr(response, "output_text", "").strip()

        request_body = _chat_completions_payload(
            model=model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        if session_logger:
            session_logger.log_api_request_body(request.label, provider, request_body)
        response = await client.chat.completions.create(**request_body)
        if session_logger and request.log_response_body:
            session_logger.log_api_response_body(
                request.label,
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
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if session_logger:
            session_logger.log_api_request_body(request.label, provider, request_body)
        response = await client.messages.create(**request_body)
        if session_logger and request.log_response_body:
            session_logger.log_api_response_body(
                request.label,
                provider,
                _response_to_loggable_body(response),
            )
        return _extract_anthropic_text(response)

    raise ValueError(f"Unsupported provider: {provider}")


async def stream_request(
    *,
    provider: str,
    model: str,
    api_key: str,
    request: LLMRequest,
    session_logger: "SessionLogger | None" = None,
) -> AsyncGenerator[str, None]:
    """Yield text deltas from the given provider."""
    if provider == "openai":
        client = AsyncOpenAI(api_key=api_key)
        if _is_gpt5_model(model):
            request_body = _responses_api_payload(
                model=model,
                messages=request.messages,
                max_tokens=request.max_tokens,
            )
            if session_logger:
                session_logger.log_api_request_body(request.label, provider, request_body)
            async with client.responses.stream(**request_body) as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta" and event.delta:
                        yield event.delta
                if session_logger and request.log_response_body:
                    final_response = await stream.get_final_response()
                    session_logger.log_api_response_body(
                        request.label,
                        provider,
                        _response_to_loggable_body(final_response),
                    )
            return

        request_body = _chat_completions_payload(
            model=model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        if session_logger:
            session_logger.log_api_request_body(request.label, provider, request_body)
        async with client.chat.completions.stream(**request_body) as stream:
            async for event in stream:
                if event.type == "content.delta" and event.delta:
                    yield event.delta
            if session_logger and request.log_response_body:
                final_completion = await stream.get_final_completion()
                session_logger.log_api_response_body(
                    request.label,
                    provider,
                    _response_to_loggable_body(final_completion),
                )
        return

    if provider == "anthropic":
        system, chat_messages = _anthropic_payload(messages)
        client = AsyncAnthropic(api_key=api_key)
        request_body = {
            "model": model,
            "system": system,
            "messages": chat_messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if session_logger:
            session_logger.log_api_request_body(request.label, provider, request_body)
        async with client.messages.stream(**request_body) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text
            if session_logger and request.log_response_body:
                final_message = await stream.get_final_message()
                session_logger.log_api_response_body(
                    request.label,
                    provider,
                    _response_to_loggable_body(final_message),
                )
        return

    raise ValueError(f"Unsupported provider: {provider}")
