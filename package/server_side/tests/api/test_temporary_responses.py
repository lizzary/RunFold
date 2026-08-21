from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from fastapi.testclient import TestClient

from runfold_server.errors import ApiError
from runfold_server.http.app import create_app
from runfold_server.http.routers.temporary_responses import TemporaryResponsesClient
from runfold_server.identity.models import AuthContext, User, VerifiedIdentity


class _IdentityService:
    def __init__(self) -> None:
        self.revalidated: list[AuthContext] = []

    def authenticate(self, token: str, request_id: str) -> VerifiedIdentity:
        if token != "registered-user-token":
            raise ApiError(401, "invalid_session", "Authentication is required")
        return VerifiedIdentity(
            context=AuthContext(
                user_id="user-1",
                session_id="session-1",
                request_id=request_id,
            ),
            user=User(
                id="user-1",
                username="pm",
                display_name="PM",
                status="active",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
            ),
        )

    def revalidate(self, context: AuthContext) -> VerifiedIdentity:
        self.revalidated.append(context)
        return self.authenticate("registered-user-token", context.request_id)


class _ResponsesClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **arguments: Any) -> dict[str, Any]:
        self.calls.append(arguments)
        return {
            "id": "resp_2",
            "object": "response",
            "status": "completed",
            "output_text": "I remember the previous turn.",
        }


def test_registered_user_can_create_contextual_response() -> None:
    identity = _IdentityService()
    responses = _ResponsesClient()
    app = create_app(
        allowed_origins=("http://localhost:3000",),
        readiness_check=lambda: True,
        identity_service=identity,  # type: ignore[arg-type]
        access_control_service=object(),  # type: ignore[arg-type]
        temporary_responses_client=responses,  # type: ignore[arg-type]
    )
    client = TestClient(app)

    missing = client.post(
        "/api/temporary/responses",
        json={"model": "test-model", "input": "hello"},
    )
    allowed = client.post(
        "/api/temporary/responses",
        headers={"Authorization": "Bearer registered-user-token"},
        json={
            "model": "test-model",
            "input": "what did I say?",
            "previous_response_id": "resp_1",
        },
    )

    assert missing.status_code == 401
    assert missing.json()["code"] == "invalid_session"
    assert allowed.status_code == 200
    assert allowed.json()["id"] == "resp_2"
    assert responses.calls == [
        {
            "model": "test-model",
            "input_text": "what did I say?",
            "previous_response_id": "resp_1",
        }
    ]
    assert len(identity.revalidated) == 1


def test_provider_client_uses_responses_protocol_without_tools() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_next",
                "object": "response",
                "status": "completed",
                "output": [],
            },
        )

    async def exercise() -> dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            client = TemporaryResponsesClient(
                http_client=http,
                base_url="https://llm.example/v1",
                api_key="provider-secret",
                max_retries=0,
            )
            return await client.create(
                model="chat-model",
                input_text="continue",
                previous_response_id="resp_first",
            )

    result = asyncio.run(exercise())

    assert result["id"] == "resp_next"
    assert len(requests) == 1
    assert str(requests[0].url) == "https://llm.example/v1/responses"
    assert requests[0].headers["Authorization"] == "Bearer provider-secret"
    assert json.loads(requests[0].content) == {
        "model": "chat-model",
        "input": "continue",
        "store": True,
        "tools": [],
        "previous_response_id": "resp_first",
    }
