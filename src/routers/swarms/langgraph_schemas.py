from pydantic import BaseModel, Field
from typing import List, Optional


class LanggraphWorkerInput(BaseModel):
    agentName: str
    agentDescription: str = ""
    systemPrompt: Optional[str] = None
    modelName: str = "gpt-4o-mini"


class LanggraphSupervisorInput(BaseModel):
    name: str = "supervisor"
    modelName: str = "gpt-4o-mini"
    systemPrompt: Optional[str] = None


class LanggraphSwarmConfigInput(BaseModel):
    supervisor: LanggraphSupervisorInput
    workers: List[LanggraphWorkerInput] = Field(..., min_length=1)
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
    prompt: Optional[str] = None  # Required when stateToken is absent (first turn)
    stateToken: Optional[str] = None  # Returned by previous response; send to get next speaker


class SwarmTtsNextTurnResponse(BaseModel):
    agentName: str
    content: str
    stateToken: Optional[str] = None  # Send back after playback to get next turn
    done: bool  # True when no more speakers for this user message
