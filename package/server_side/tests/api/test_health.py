from __future__ import annotations

import re

from fastapi.testclient import TestClient

from runfold_server.http.app import create_app


def test_live_and_ready_health() -> None:
    client = TestClient(
        create_app(
            allowed_origins=("http://localhost:3000",),
            readiness_check=lambda: True,
        )
    )

    assert client.get("/health/live").json() == {"status": "live"}
    assert client.get("/health/ready").json() == {"status": "ready"}


def test_not_ready_uses_safe_unified_error() -> None:
    client = TestClient(
        create_app(
            allowed_origins=("http://localhost:3000",),
            readiness_check=lambda: False,
        )
    )

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "code": "service_not_ready",
        "message": "Service is not ready",
        "details": {},
        "request_id": response.headers["X-Request-ID"],
    }


def test_valid_request_id_is_echoed_and_invalid_one_is_replaced() -> None:
    client = TestClient(
        create_app(
            allowed_origins=("http://localhost:3000",),
            readiness_check=lambda: True,
        )
    )

    accepted = client.get("/health/live", headers={"X-Request-ID": "client-request_1"})
    replaced = client.get("/health/live", headers={"X-Request-ID": "bad request id"})

    assert accepted.headers["X-Request-ID"] == "client-request_1"
    assert replaced.headers["X-Request-ID"] != "bad request id"
    assert re.fullmatch(r"[0-9a-f]{32}", replaced.headers["X-Request-ID"])


def test_unknown_route_uses_unified_error() -> None:
    client = TestClient(
        create_app(
            allowed_origins=("http://localhost:3000",),
            readiness_check=lambda: True,
        )
    )

    response = client.get("/not-present")

    assert response.status_code == 404
    assert response.json()["code"] == "route_not_found"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_unhandled_exception_is_safe() -> None:
    app = create_app(
        allowed_origins=("http://localhost:3000",),
        readiness_check=lambda: True,
    )

    @app.get("/explode")
    def explode() -> None:
        raise RuntimeError("password=must-not-reach-client")

    response = TestClient(app, raise_server_exceptions=False).get("/explode")

    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert "must-not-reach-client" not in response.text


def test_openapi_is_unversioned_and_contains_no_future_routes() -> None:
    app = create_app(
        allowed_origins=("http://localhost:3000",),
        readiness_check=lambda: True,
    )

    schema = app.openapi()

    assert schema["info"]["version"] == "unversioned"
    assert set(schema["paths"]) == {"/health/live", "/health/ready"}


def test_cors_allows_only_configured_origin() -> None:
    client = TestClient(
        create_app(
            allowed_origins=("https://allowed.example",),
            readiness_check=lambda: True,
        )
    )

    allowed = client.options(
        "/health/live",
        headers={
            "Origin": "https://allowed.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    denied = client.options(
        "/health/live",
        headers={
            "Origin": "https://denied.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert allowed.headers["access-control-allow-origin"] == "https://allowed.example"
    assert "access-control-allow-origin" not in denied.headers
    assert denied.json()["code"] == "cors_request_denied"
    assert denied.json()["request_id"] == denied.headers["X-Request-ID"]
