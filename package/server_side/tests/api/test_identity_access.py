from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from runfold_server.access_control.capabilities import (
    ALL_CAPABILITIES,
    IDENTITY_USER_MANAGE,
    IDENTITY_USER_READ,
    SYSTEM_ADMIN_ROLE_ID,
)
from runfold_server.bootstrap import bootstrap
from runfold_server.config import load_settings
from runfold_server.storage.sqlite import connect

ADMIN_PASSWORD = "correct horse battery staple"


@pytest.fixture
def m1_client(admin_environment: dict[str, str]) -> tuple[TestClient, Path]:
    settings = load_settings(admin_environment)
    return TestClient(bootstrap(settings)), settings.data_dir / "runfold.sqlite3"


def _login(client: TestClient, username: str, password: str) -> tuple[str, dict[str, object]]:
    response = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    return payload["token"], payload["user"]


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _create_user(
    client: TestClient,
    admin_token: str,
    *,
    username: str,
    password: str = "another correct password",
) -> dict[str, object]:
    response = client.post(
        "/api/access/users",
        headers=_headers(admin_token),
        json={
            "username": username,
            "display_name": username.title(),
            "password": password,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _create_role(client: TestClient, admin_token: str, name: str) -> dict[str, object]:
    response = client.post(
        "/api/access/roles",
        headers=_headers(admin_token),
        json={"name": name, "description": "test role"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_bootstrap_login_opaque_session_and_authentication(
    m1_client: tuple[TestClient, Path],
) -> None:
    client, database = m1_client
    token, admin = _login(client, "ADMIN", ADMIN_PASSWORD)

    me = client.get("/api/auth/me", headers=_headers(token))
    missing = client.get("/api/auth/me")
    invalid = client.get("/api/auth/me", headers=_headers("not-a-session"))

    assert me.status_code == 200
    assert me.json() == admin
    assert missing.status_code == invalid.status_code == 401
    assert missing.json()["code"] == invalid.json()["code"] == "invalid_session"
    assert "role" not in me.text.lower()
    assert "capabilit" not in me.text.lower()

    with connect(database) as connection:
        user_row = connection.execute(
            "SELECT password_hash FROM users WHERE username = 'admin'"
        ).fetchone()
        session_row = connection.execute(
            "SELECT token_hash FROM auth_sessions WHERE user_id = ?",
            (admin["id"],),
        ).fetchone()
    assert user_row[0].startswith("$argon2id$")
    assert session_row[0] == hashlib.sha256(token.encode()).hexdigest()
    assert token not in session_row[0]


def test_expired_revoked_and_hashed_session_values_are_not_credentials(
    m1_client: tuple[TestClient, Path],
) -> None:
    client, database = m1_client
    token, _ = _login(client, "admin", ADMIN_PASSWORD)
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    assert client.get("/api/auth/me", headers=_headers(token_hash)).status_code == 401
    with connect(database) as connection:
        connection.execute(
            "UPDATE auth_sessions SET expires_at = '1970-01-01T00:00:00+00:00' "
            "WHERE token_hash = ?",
            (token_hash,),
        )
    assert client.get("/api/auth/me", headers=_headers(token)).status_code == 401

    fresh, _ = _login(client, "admin", ADMIN_PASSWORD)
    with connect(database) as connection:
        connection.execute(
            "UPDATE auth_sessions SET revoked_at = '1970-01-01T00:00:00+00:00' "
            "WHERE token_hash = ?",
            (hashlib.sha256(fresh.encode()).hexdigest(),),
        )
    assert client.get("/api/auth/me", headers=_headers(fresh)).status_code == 401


def test_login_failure_is_generic_and_audited_without_secrets(
    m1_client: tuple[TestClient, Path],
) -> None:
    client, database = m1_client
    secret = "wrong password value"

    missing = client.post(
        "/api/auth/login", json={"username": "missing", "password": secret}
    )
    wrong = client.post(
        "/api/auth/login", json={"username": "admin", "password": secret}
    )

    assert missing.status_code == wrong.status_code == 401
    assert missing.json()["code"] == wrong.json()["code"] == "invalid_credentials"
    assert secret not in missing.text + wrong.text
    with connect(database) as connection:
        events = connection.execute(
            "SELECT action, decision, reason, details_json FROM audit_events ORDER BY id"
        ).fetchall()
    serialized = json.dumps([tuple(row) for row in events])
    assert sum(row[0] == "auth.login" and row[1] == "denied" for row in events) == 2
    assert secret not in serialized


def test_password_change_revokes_every_session(
    m1_client: tuple[TestClient, Path],
) -> None:
    client, _ = m1_client
    first, _ = _login(client, "admin", ADMIN_PASSWORD)
    second, _ = _login(client, "admin", ADMIN_PASSWORD)
    new_password = "new correct horse password"

    response = client.put(
        "/api/auth/password",
        headers=_headers(first),
        json={"current_password": ADMIN_PASSWORD, "new_password": new_password},
    )

    assert response.status_code == 204
    assert client.get("/api/auth/me", headers=_headers(first)).status_code == 401
    assert client.get("/api/auth/me", headers=_headers(second)).status_code == 401
    assert client.post(
        "/api/auth/login",
        json={"username": "admin", "password": ADMIN_PASSWORD},
    ).status_code == 401
    _login(client, "admin", new_password)


def test_permissions_are_reloaded_on_the_next_request(
    m1_client: tuple[TestClient, Path],
) -> None:
    client, _ = m1_client
    admin_token, _ = _login(client, "admin", ADMIN_PASSWORD)
    user = _create_user(client, admin_token, username="alice")
    role = _create_role(client, admin_token, "user_directory_reader")
    role_url = f"/api/access/roles/{role['id']}/capabilities"
    assignment_url = f"/api/access/users/{user['id']}/roles"
    assert client.put(
        role_url,
        headers=_headers(admin_token),
        json={"capability_codes": [IDENTITY_USER_READ]},
    ).status_code == 200
    assert client.put(
        assignment_url,
        headers=_headers(admin_token),
        json={"role_ids": [role["id"]]},
    ).status_code == 200
    assert client.get(assignment_url, headers=_headers(admin_token)).json()[
        "role_ids"
    ] == [role["id"]]
    alice_token, _ = _login(client, "alice", "another correct password")

    assert client.get("/api/access/users", headers=_headers(alice_token)).status_code == 200
    assert client.put(
        role_url,
        headers=_headers(admin_token),
        json={"capability_codes": []},
    ).status_code == 200
    denied = client.get("/api/access/users", headers=_headers(alice_token))
    assert denied.status_code == 403
    assert denied.json()["code"] == "permission_denied"


def test_root_capabilities_require_direct_protected_membership_even_if_database_is_tampered(
    m1_client: tuple[TestClient, Path],
) -> None:
    client, database = m1_client
    admin_token, _ = _login(client, "admin", ADMIN_PASSWORD)
    user = _create_user(client, admin_token, username="mallory")
    role = _create_role(client, admin_token, "ordinary_admin_like_role")
    restricted = client.put(
        f"/api/access/roles/{role['id']}/capabilities",
        headers=_headers(admin_token),
        json={"capability_codes": [IDENTITY_USER_MANAGE]},
    )
    assert restricted.status_code == 403
    assert restricted.json()["code"] == "root_capability_restricted"

    assert client.put(
        f"/api/access/users/{user['id']}/roles",
        headers=_headers(admin_token),
        json={"role_ids": [role["id"]]},
    ).status_code == 200
    with connect(database) as connection:
        connection.execute(
            "INSERT INTO role_capabilities (role_id, capability_code) VALUES (?, ?)",
            (role["id"], IDENTITY_USER_MANAGE),
        )
    mallory_token, _ = _login(client, "mallory", "another correct password")
    denied = client.post(
        "/api/access/users",
        headers=_headers(mallory_token),
        json={
            "username": "should-not-exist",
            "display_name": "Denied",
            "password": "valid but denied password",
        },
    )
    assert denied.status_code == 403
    assert denied.json()["code"] == "permission_denied"


def test_protected_role_and_last_active_admin_cannot_be_destroyed(
    m1_client: tuple[TestClient, Path],
) -> None:
    client, _ = m1_client
    admin_token, admin = _login(client, "admin", ADMIN_PASSWORD)
    protected_url = f"/api/access/roles/{SYSTEM_ADMIN_ROLE_ID}"

    assert client.patch(
        protected_url,
        headers=_headers(admin_token),
        json={"name": "renamed"},
    ).json()["code"] == "protected_role"
    assert client.delete(protected_url, headers=_headers(admin_token)).json()[
        "code"
    ] == "protected_role"
    assert client.put(
        f"{protected_url}/capabilities",
        headers=_headers(admin_token),
        json={"capability_codes": []},
    ).json()["code"] == "protected_role"

    disable = client.patch(
        f"/api/access/users/{admin['id']}",
        headers=_headers(admin_token),
        json={"status": "disabled"},
    )
    remove_role = client.put(
        f"/api/access/users/{admin['id']}/roles",
        headers=_headers(admin_token),
        json={"role_ids": []},
    )
    assert disable.status_code == remove_role.status_code == 409
    assert disable.json()["code"] == remove_role.json()["code"] == "last_system_admin"
    assert client.get("/api/auth/me", headers=_headers(admin_token)).status_code == 200


def test_admin_reset_revokes_target_sessions_and_user_disable_is_immediate(
    m1_client: tuple[TestClient, Path],
) -> None:
    client, _ = m1_client
    admin_token, _ = _login(client, "admin", ADMIN_PASSWORD)
    user = _create_user(client, admin_token, username="bob")
    old_password = "another correct password"
    bob_token, _ = _login(client, "bob", old_password)
    new_password = "replacement correct password"

    reset = client.put(
        f"/api/access/users/{user['id']}/password",
        headers=_headers(admin_token),
        json={"new_password": new_password},
    )
    assert reset.status_code == 204
    assert client.get("/api/auth/me", headers=_headers(bob_token)).status_code == 401
    bob_token, _ = _login(client, "bob", new_password)

    disabled = client.patch(
        f"/api/access/users/{user['id']}",
        headers=_headers(admin_token),
        json={"status": "disabled"},
    )
    assert disabled.status_code == 200
    assert client.get("/api/auth/me", headers=_headers(bob_token)).status_code == 401
    assert client.post(
        "/api/auth/login", json={"username": "bob", "password": new_password}
    ).status_code == 401


def test_access_routes_reject_unknown_fields_and_paginate(
    m1_client: tuple[TestClient, Path],
) -> None:
    client, _ = m1_client
    admin_token, _ = _login(client, "admin", ADMIN_PASSWORD)
    headers = _headers(admin_token)

    capabilities = client.get("/api/access/capabilities?limit=5&offset=2", headers=headers)
    roles = client.get("/api/access/roles?limit=2&offset=1", headers=headers)
    users = client.get("/api/access/users?limit=10&offset=0", headers=headers)
    invalid = client.post(
        "/api/access/roles",
        headers=headers,
        json={"name": "extra", "description": "", "is_protected": True},
    )

    assert capabilities.status_code == roles.status_code == users.status_code == 200
    assert capabilities.json()["total"] == len(ALL_CAPABILITIES)
    assert len(capabilities.json()["items"]) == 5
    assert len(roles.json()["items"]) == 2
    assert users.json()["total"] == 1
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "invalid_request"


def test_every_access_route_denies_a_session_without_required_capabilities(
    m1_client: tuple[TestClient, Path],
) -> None:
    client, _ = m1_client
    admin_token, admin = _login(client, "admin", ADMIN_PASSWORD)
    user = _create_user(client, admin_token, username="unprivileged")
    role = _create_role(client, admin_token, "denial_target")
    token, _ = _login(client, "unprivileged", "another correct password")
    headers = _headers(token)
    requests = [
        ("GET", "/api/access/capabilities", None),
        ("GET", "/api/access/users", None),
        ("POST", "/api/access/users", {
            "username": "blocked-user",
            "display_name": "Blocked",
            "password": "blocked valid password",
        }),
        ("GET", f"/api/access/users/{admin['id']}", None),
        ("PATCH", f"/api/access/users/{user['id']}", {"display_name": "Blocked"}),
        ("PUT", f"/api/access/users/{user['id']}/password", {
            "new_password": "blocked replacement password",
        }),
        ("PUT", f"/api/access/users/{user['id']}/roles", {"role_ids": []}),
        ("GET", f"/api/access/users/{user['id']}/roles", None),
        ("GET", "/api/access/roles", None),
        ("POST", "/api/access/roles", {"name": "blocked-role", "description": ""}),
        ("GET", f"/api/access/roles/{role['id']}", None),
        ("PATCH", f"/api/access/roles/{role['id']}", {"description": "Blocked"}),
        ("DELETE", f"/api/access/roles/{role['id']}", None),
        ("PUT", f"/api/access/roles/{role['id']}/capabilities", {
            "capability_codes": [],
        }),
    ]

    for method, path, body in requests:
        response = client.request(method, path, headers=headers, json=body)
        assert response.status_code == 403, (method, path, response.text)
        assert response.json()["code"] == "permission_denied"


def test_allowed_role_crud_user_restore_and_logout(
    m1_client: tuple[TestClient, Path],
) -> None:
    client, _ = m1_client
    token, _ = _login(client, "admin", ADMIN_PASSWORD)
    headers = _headers(token)
    user = _create_user(client, token, username="lifecycle")
    role = _create_role(client, token, "lifecycle_role")

    updated_role = client.patch(
        f"/api/access/roles/{role['id']}",
        headers=headers,
        json={"name": "lifecycle_role_renamed", "description": "updated"},
    )
    role_detail = client.get(f"/api/access/roles/{role['id']}", headers=headers)
    disabled = client.patch(
        f"/api/access/users/{user['id']}",
        headers=headers,
        json={"status": "disabled"},
    )
    restored = client.patch(
        f"/api/access/users/{user['id']}",
        headers=headers,
        json={"status": "active"},
    )
    user_detail = client.get(f"/api/access/users/{user['id']}", headers=headers)
    deleted = client.delete(f"/api/access/roles/{role['id']}", headers=headers)
    logout = client.post("/api/auth/logout", headers=headers)

    assert updated_role.status_code == role_detail.status_code == 200
    assert role_detail.json()["name"] == "lifecycle_role_renamed"
    assert disabled.json()["status"] == "disabled"
    assert restored.json()["status"] == user_detail.json()["status"] == "active"
    assert deleted.status_code == logout.status_code == 204
    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_bootstrapped_openapi_contains_m4_and_no_future_routes(
    m1_client: tuple[TestClient, Path],
) -> None:
    client, _ = m1_client
    paths = set(client.app.openapi()["paths"])

    assert "/api/auth/login" in paths
    assert "/api/access/users/{user_id}/roles" in paths
    assert "/api/access/roles/{role_id}/capabilities" in paths
    assert "/api/rag/documents" in paths
    assert "/api/rag/documents/{document_id}/acl" in paths
    assert "/api/rag/search" in paths
    assert "/api/usage/me" in paths
    assert "/api/usage/users/{user_id}/limits" in paths
    assert "/api/security/audit" in paths
    assert not any(
        forbidden in path
        for path in paths
            for forbidden in (
                "/agent",
            "/tools",
            "/skills",
        )
    )
