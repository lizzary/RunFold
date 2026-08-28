from __future__ import annotations

from conftest import ConfigFile
from fastapi.testclient import TestClient

from runfold_server.bootstrap import bootstrap
from runfold_server.errors import ApiError
from runfold_server.http.app import create_app
from runfold_server.identity.models import AuthContext, User, VerifiedIdentity
from runfold_server.runtime.models import AgentRunResult


class _Identity:
    def authenticate(self, token: str, request_id: str) -> VerifiedIdentity:
        if token != "agent-token":
            raise ApiError(401, "invalid_session", "Authentication is required")
        return VerifiedIdentity(
            context=AuthContext(
                user_id="user-1",
                session_id="session-1",
                request_id=request_id,
            ),
            user=User(
                id="user-1",
                username="agent-user",
                display_name="Agent User",
                status="active",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
            ),
        )


class _Runtime:
    def __init__(self) -> None:
        self.inputs: list[tuple[str, str | None]] = []

    async def run(
        self,
        actor: VerifiedIdentity,
        user_input: str,
        *,
        thinking_level: str | None = None,
    ) -> AgentRunResult:
        assert actor.user_id == "user-1"
        self.inputs.append((user_input, thinking_level))
        return AgentRunResult(
            answer="The /root decision.",
            reasoning_content="The root checked every report.",
            thinking_level=thinking_level,
            agents_created=2,
            max_depth_reached=1,
        )


def test_agent_route_exposes_only_root_answer_and_safe_team_metrics() -> None:
    runtime = _Runtime()
    app = create_app(
        allowed_origins=("http://localhost:3000",),
        readiness_check=lambda: True,
        identity_service=_Identity(),  # type: ignore[arg-type]
        access_control_service=object(),  # type: ignore[arg-type]
        runtime_service=runtime,  # type: ignore[arg-type]
    )
    client = TestClient(app)

    missing = client.post("/api/agent/runs", json={"input": "decide"})
    allowed = client.post(
        "/api/agent/runs",
        headers={"Authorization": "Bearer agent-token"},
        json={"input": "decide", "thinking_level": " HIGH "},
    )

    assert missing.status_code == 401
    assert allowed.status_code == 200
    assert allowed.json() == {
        "answer": "The /root decision.",
        "reasoning_content": "The root checked every report.",
        "thinking_level": "high",
        "agents_created": 2,
        "max_depth_reached": 1,
    }
    assert runtime.inputs == [("decide", "high")]
    assert "agent_path" not in allowed.text
    assert "private_task_history" not in allowed.text


def test_agent_run_requires_capability_before_any_provider_call(
    admin_config: ConfigFile,
) -> None:
    client = TestClient(bootstrap(admin_config.load()))
    admin = _login(client, "admin", "correct horse battery staple")
    created = client.post(
        "/api/access/users",
        headers=admin,
        json={
            "username": "no-agent-role",
            "display_name": "No Agent Role",
            "password": "no agent password 123",
        },
    )
    assert created.status_code == 201, created.text
    user = _login(client, "no-agent-role", "no agent password 123")

    denied = client.post(
        "/api/agent/runs",
        headers=user,
        json={"input": "This must not reach the provider."},
    )

    assert denied.status_code == 403
    assert denied.json()["code"] == "permission_denied"
    audit = client.get(
        "/api/security/audit",
        headers=admin,
        params={"action": "agent.run", "limit": 10},
    )
    assert audit.status_code == 200
    assert audit.json()["total"] == 1
    assert audit.json()["items"][0]["decision"] == "denied"
    assert audit.json()["items"][0]["reason"] == "permission_denied"


def _login(client: TestClient, username: str, password: str) -> dict[str, str]:
    response = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}
