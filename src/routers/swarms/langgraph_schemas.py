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
