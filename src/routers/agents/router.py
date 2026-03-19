"""
Agents router for the completion API.

Aggregates all agent-related endpoints:
  - POST /{agent_id}/stream      — streaming LLM completion (SSE)
  - POST /{agent_id}/completion  — non-streaming, returns full output as JSON
  - GET  /{agent_id}             — read agent configuration
"""
from fastapi import APIRouter

from .stream import router as stream_router
from .completion import router as completion_router
from .get import router as get_router

router = APIRouter()

router.include_router(stream_router)
router.include_router(completion_router)
router.include_router(get_router)
