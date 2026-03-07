from pydantic import BaseModel, Field
from typing import List, Optional


class SwarmWorkerInput(BaseModel):
    agentName: str
    agentDescription: str = ""
    systemPrompt: Optional[str] = None
    modelName: str = "gpt-4o-mini"


class SwarmDirectorInput(BaseModel):
    name: str = "Director"
    modelName: str = "gpt-4o-mini"
    systemPrompt: Optional[str] = None


class SwarmConfigInput(BaseModel):
    director: SwarmDirectorInput
    workers: List[SwarmWorkerInput] = Field(..., min_length=1)
    maxLoops: int = 1


class SwarmCompletionRequest(BaseModel):
    prompt: str
    sessionId: str
    swarm: SwarmConfigInput
