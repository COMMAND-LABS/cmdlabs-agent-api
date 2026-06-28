"""PDF to FAQ router.

Aggregates the one-shot PDF -> FAQ generation endpoint:
  - POST /generate  — generate Q&A pairs from an uploaded PDF (non-streaming JSON)
"""
from fastapi import APIRouter

from .generate import router as generate_router

router = APIRouter()

router.include_router(generate_router)
