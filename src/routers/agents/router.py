"""
Agents router for the completion API.

Aggregates all agent-related endpoints:
  - POST /{agent_id}/stream      — streaming LLM completion (SSE)
  - GET  /{agent_id}             — read agent configuration
"""
from fastapi import APIRouter

from .get import router as get_router
from .stream import router as stream_router

router = APIRouter()

router.include_router(stream_router)
router.include_router(get_router)
