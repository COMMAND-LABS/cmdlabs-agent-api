"""
LLM factory for creating language model instances based on agent config.

Supports:
- OpenAI (gpt-4o-mini, gpt-4o, etc.)
- Anthropic (claude-3-5-sonnet, claude-3-5-haiku, etc.)
- Google (gemini-2.0-flash, gemini-1.5-pro, etc.)
- Ollama (llama3.2, mistral, etc.)
"""
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_MODEL_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-mini",
}


def get_model_config(agent_config: dict[str, Any]) -> dict[str, str]:
    """
    Extract model configuration from a v4 agent config.
    Returns the configured model if present, otherwise the default.
    """
    config_data = agent_config.get('data', {})
    model_config = config_data.get('model')
    if model_config:
        resolved = {
            "provider": model_config.get('provider', 'openai'),
            "model": model_config.get('model', 'gpt-4o-mini'),
        }
        # Optional explicit credential binding for turn completions. When present
        # it pins the exact credential (no drift); when absent the funding
        # account's default for the provider type is resolved at runtime.
        if model_config.get('credentialId') is not None:
            resolved["credentialId"] = model_config["credentialId"]
        return resolved
    return DEFAULT_MODEL_CONFIG.copy()


def create_llm(
    model_config: dict[str, str],
    credentials: dict[str, str],
    temperature: float = 0,
) -> tuple[BaseChatModel, str]:
    """
    Create a streaming LangChain LLM instance based on model configuration.

    Args:
        model_config: Dict with 'provider' and 'model' keys
        credentials: Dict mapping provider names to API keys
                    e.g., {'openai': 'sk-...', 'anthropic': 'sk-ant-...'}
        temperature: Model temperature (0-1)

    Returns:
        Tuple of (LLM instance, provider name)

    Raises:
        ValueError: If provider is not supported or credentials are missing
    """
    provider = model_config.get('provider', 'openai')
    model = model_config.get('model', 'gpt-4o-mini')

    if provider == 'openai':
        return _create_openai_llm(model, credentials, temperature), provider
    if provider == 'anthropic':
        return _create_anthropic_llm(model, credentials, temperature), provider
    if provider == 'google':
        return _create_google_llm(model, credentials, temperature), provider
    if provider == 'ollama':
        return _create_ollama_llm(model, temperature), provider
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _create_openai_llm(
    model: str,
    credentials: dict[str, str],
    temperature: float,
) -> BaseChatModel:
    """Create OpenAI LLM instance."""
    from langchain_openai import ChatOpenAI

    api_key = credentials.get('openai')
    if not api_key:
        raise ValueError("OpenAI API key not found. Please add your OpenAI API key in account settings.")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        streaming=True,
        temperature=temperature,
        stream_usage=True,
        model_kwargs={"parallel_tool_calls": False},
    )


def _create_anthropic_llm(
    model: str,
    credentials: dict[str, str],
    temperature: float,
) -> BaseChatModel:
    """Create Anthropic LLM instance."""
    from langchain_anthropic import ChatAnthropic

    api_key = credentials.get('anthropic')
    if not api_key:
        raise ValueError("Anthropic API key not found. Please add your Anthropic API key in account settings.")

    return ChatAnthropic(
        model=model,
        api_key=api_key,
        streaming=True,
        temperature=temperature,
    )


def _create_google_llm(
    model: str,
    credentials: dict[str, str],
    temperature: float,
) -> BaseChatModel:
    """Create Google Gemini LLM instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = credentials.get('google')
    if not api_key:
        raise ValueError("Google Gemini API key not found. Please add your Google Gemini API key in account settings.")

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        streaming=True,
        temperature=temperature,
    )


def _create_ollama_llm(
    model: str,
    temperature: float,
) -> BaseChatModel:
    """Create Ollama LLM instance."""
    import os

    from langchain_ollama import ChatOllama

    # Ollama base URL can be configured via environment variable
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    return ChatOllama(
        model=model,
        base_url=base_url,
        streaming=True,
        temperature=temperature,
    )


def get_required_credential_type(provider: str) -> str | None:
    """
    Get the ServiceName credential type required for a provider.

    Args:
        provider: The LLM provider name

    Returns:
        ServiceName enum value, or None if no credential needed
    """
    from src.db.service_name import ServiceName

    provider_to_credential = {
        'openai': ServiceName.OPENAI_API_KEY,
        'anthropic': ServiceName.ANTHROPIC_API_KEY,
        'google': ServiceName.GOOGLE_GEMINI_API_KEY,
        'ollama': None,  # Ollama is self-hosted, no API key needed
    }

    return provider_to_credential.get(provider)
