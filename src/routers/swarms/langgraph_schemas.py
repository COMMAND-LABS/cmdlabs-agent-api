
from pydantic import BaseModel, Field


class LanggraphWorkerInput(BaseModel):
    agentName: str
    agentDescription: str = ""
    systemPrompt: str | None = None
    modelName: str = "gpt-4o-mini"


class LanggraphSupervisorInput(BaseModel):
    name: str = "supervisor"
    modelName: str = "gpt-4o-mini"
    systemPrompt: str | None = None


class LanggraphSwarmConfigInput(BaseModel):
    supervisor: LanggraphSupervisorInput
    workers: list[LanggraphWorkerInput] = Field(..., min_length=1)
    outputMode: str = "last_message"


class LanggraphSwarmCompletionRequest(BaseModel):
    prompt: str
    sessionId: str
    swarm: LanggraphSwarmConfigInput


# --- TTS next-turn (one agent per request) ---


class SwarmTtsNextTurnRequest(BaseModel):
    """Request for one agent turn. Send prompt on first turn; send stateToken after playback to get next turn."""
    sessionId: str
    swarm: LanggraphSwarmConfigInput
    prompt: str | None = None  # Required when stateToken is absent (first turn)
    stateToken: str | None = None  # Returned by previous response; send to get next speaker


class SwarmTtsNextTurnResponse(BaseModel):
    agentName: str
    content: str
    stateToken: str | None = None  # Send back after playback to get next turn
    done: bool  # True when no more speakers for this user message
