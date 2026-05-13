"""
Health check.
"""
from __future__ import annotations

from fastapi import APIRouter

from ..models.factory import create_model

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "default_model": create_model(None).get_info()}
