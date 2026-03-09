from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.multi_agent.session_logger import SessionLogger


@dataclass(frozen=True)
class ConversationEntry:
    role: str
    content: str
    agent_name: Optional[str] = None


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    model: str
    system_prompt: str


@dataclass(frozen=True)
class LLMRequest:
    messages: list[dict[str, str]]
    temperature: float
    max_tokens: int
    label: str
    log_response_body: bool = False


@dataclass(frozen=True)
class RouterDecision:
    next_speakers: list[str]
    reason: str = ""


@dataclass(frozen=True)
class TurnState:
    session_id: str
    response_count: int
    last_message_id: Optional[int]
    swarm_hash: str


@dataclass
class PreparedTurnContext:
    db: Any
    provider: str
    api_key: str
    supervisor_model: str
    supervisor_prompt: Optional[str]
    agent_definitions: dict[str, AgentDefinition]
    agent_list: list[dict[str, str]]
    history: list[ConversationEntry]
    chat_session_id: int
    state: TurnState
    session_logger: SessionLogger


@dataclass(frozen=True)
class TurnResult:
    agent_name: str
    content: str
    state_token: Optional[str]
    done: bool
    route_reason: str


@dataclass(frozen=True)
class StreamEvent:
    event: str
    data: dict[str, Any]
