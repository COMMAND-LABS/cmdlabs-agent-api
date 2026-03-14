"""
Shared Pydantic models for the agents router.
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any


class AgentResponse(BaseModel):
    id: int
    name: str
    config: Optional[Dict[str, Any]] = None
    is_owner: Optional[bool] = None

    class Config:
        from_attributes = True
