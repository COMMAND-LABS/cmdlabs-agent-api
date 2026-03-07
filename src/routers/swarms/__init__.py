from fastapi import APIRouter

from .swarm_completion import router as _swarm_router
from .langgraph_swarm import router as _langgraph_router

router = APIRouter()
router.include_router(_swarm_router)
router.include_router(_langgraph_router)
