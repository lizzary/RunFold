from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from runfold_server.access_control.service import AccessControlService
from runfold_server.errors import ApiError
from runfold_server.http.routers.access_control import create_access_control_router
from runfold_server.http.routers.auth import create_auth_router
from runfold_server.http.routers.health import create_health_router
from runfold_server.identity.service import IdentityService

_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_LOGGER = logging.getLogger("runfold_server.http")


def create_app(
    *,
    allowed_origins: tuple[str, ...],
    readiness_check: Callable[[], bool],
    identity_service: IdentityService | None = None,
    access_control_service: AccessControlService | None = None,
) -> FastAPI:
    app = FastAPI(title="RunFold Server", version="unversioned")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_observability(request: Request, call_next: Callable) -> Response:
        request_id = _accepted_request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as error:
            request.state.error_code = "internal_error"
            _LOGGER.error(
                "unhandled_request_exception",
                extra={"request_id": request_id},
                exc_info=(type(error), error, error.__traceback__),
            )
            response = _error_response(
                request_id=request_id,
                status_code=500,
                code="internal_error",
                message="An internal error occurred",
            )
        if _is_denied_cors_preflight(request, response):
            request.state.error_code = "cors_request_denied"
            response = _error_response(
                request_id=request_id,
                status_code=400,
                code="cors_request_denied",
                message="CORS request denied",
            )
        response.headers["X-Request-ID"] = request_id
        route = request.scope.get("route")
        route_template = getattr(route, "path", "unmatched")
        _LOGGER.info(
            "http_request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "route": route_template,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1_000, 3),
                "actor_id": getattr(request.state, "actor_id", None),
                "error_code": getattr(request.state, "error_code", None),
            },
        )
        return response

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, error: ApiError) -> JSONResponse:
        request.state.error_code = error.code
        return _error_response(
            request_id=_request_id(request),
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            details=error.details,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        request.state.error_code = "invalid_request"
        fields = [
            {
                "field": ".".join(str(part) for part in item["loc"]),
                "message": item["msg"],
            }
            for item in error.errors()
        ]
        return _error_response(
            request_id=_request_id(request),
            status_code=422,
            code="invalid_request",
            message="Request validation failed",
            details={"fields": fields},
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, error: StarletteHTTPException) -> JSONResponse:
        code = "route_not_found" if error.status_code == 404 else "http_error"
        message = "Route not found" if error.status_code == 404 else "HTTP request failed"
        request.state.error_code = code
        return _error_response(
            request_id=_request_id(request),
            status_code=error.status_code,
            code=code,
            message=message,
        )

    app.include_router(create_health_router(readiness_check))
    if (identity_service is None) != (access_control_service is None):
        raise ValueError("Identity and access-control services must be installed together")
    if identity_service is not None and access_control_service is not None:
        app.include_router(create_auth_router(identity_service))
        app.include_router(
            create_access_control_router(identity_service, access_control_service)
        )
    return app


def _accepted_request_id(candidate: str | None) -> str:
    if candidate is not None and _REQUEST_ID.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", uuid.uuid4().hex)


def _is_denied_cors_preflight(request: Request, response: Response) -> bool:
    return (
        request.method == "OPTIONS"
        and "origin" in request.headers
        and "access-control-request-method" in request.headers
        and response.status_code == 400
    )


def _error_response(
    *,
    request_id: str,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, object] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers={"WWW-Authenticate": "Bearer"} if status_code == 401 else None,
        content={
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id,
        },
    )
