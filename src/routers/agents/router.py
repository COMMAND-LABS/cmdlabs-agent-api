"""
Agents router for the completion API.

Aggregates all agent-related endpoints:
  - POST /{agent_id}/completion  — streaming LLM completion
  - GET  /{agent_id}             — read agent configuration
"""
from fastapi import APIRouter

from .completion import router as completion_router
from .get import router as get_router

router = APIRouter()

router.include_router(completion_router)
router.include_router(get_router)
