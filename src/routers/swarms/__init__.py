from fastapi import APIRouter

from .langgraph_swarm import router as _langgraph_router

router = APIRouter()
router.include_router(_langgraph_router)

# NOTE: The in-house swarm router (swarm_completion) is disabled because
# src/swarm/ is missing its source files. Re-enable once restored:
# from .swarm_completion import router as _swarm_router
# router.include_router(_swarm_router)
