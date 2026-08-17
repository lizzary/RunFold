from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter

from runfold_server.errors import ApiError


def create_health_router(readiness_check: Callable[[], bool]) -> APIRouter:
    router = APIRouter(prefix="/health", tags=["health"])

    @router.get("/live")
    def live() -> dict[str, str]:
        return {"status": "live"}

    @router.get("/ready")
    def ready() -> dict[str, str]:
        if not readiness_check():
            raise ApiError(503, "service_not_ready", "Service is not ready")
        return {"status": "ready"}

    return router

