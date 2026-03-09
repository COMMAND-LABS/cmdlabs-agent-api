from fastapi import APIRouter

from .tts_next_turn import router as _tts_next_turn_router

router = APIRouter()
router.include_router(_tts_next_turn_router)
