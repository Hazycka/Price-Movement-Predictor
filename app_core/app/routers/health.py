"""
Health check.
"""
from __future__ import annotations

from fastapi import APIRouter

from ..models.factory import get_inference_model

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "default_model": get_inference_model(None).get_info()}
