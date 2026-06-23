"""
Shared Pydantic models for the agents router.
"""
from typing import Any

from pydantic import BaseModel


class AgentResponse(BaseModel):
    id: int
    name: str
    config: dict[str, Any] | None = None
    is_owner: bool | None = None

    class Config:
        from_attributes = True
