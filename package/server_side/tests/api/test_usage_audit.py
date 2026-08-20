from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from conftest import ConfigFile
from fastapi.testclient import TestClient

from runfold_server.bootstrap import bootstrap

READER_ROLE_ID = "00000000-0000-4000-8000-000000000004"


def test_usage_limits_are_aggregate_and_audit_is_system_admin_only(
    admin_config: ConfigFile,
) -> None:
    settings = admin_config.load()
    client = TestClient(bootstrap(settings))
    admin = _login(client, "admin", "correct horse battery staple")
    user = client.post(
        "/api/access/users",
        headers=admin,
        json={
            "username": "usage-reader",
            "display_name": "Usage Reader",
            "password": "reader password 123",
        },
    ).json()
    assigned = client.put(
        f"/api/access/users/{user['id']}/roles",
        headers=admin,
        json={"role_ids": [READER_ROLE_ID]},
    )
    assert assigned.status_code == 200
    _insert_hidden_billed_document(settings.data_dir / "runfold.sqlite3", user["id"])
    reader = _login(client, "usage-reader", "reader password 123")

    own = client.get("/api/usage/me", headers=reader)
    assert own.status_code == 200
    payload = own.json()
    assert payload["documents"]["current"] == 1
    assert payload["storage_bytes"]["current"] == 17
    serialized = json.dumps(payload)
    assert "hidden-billing-document" not in serialized
    assert "title" not in serialized
    assert client.get(f"/api/usage/users/{user['id']}", headers=reader).status_code == 403
    assert client.get("/api/security/audit", headers=reader).status_code == 403

    replaced = client.put(
        f"/api/usage/users/{user['id']}/limits",
        headers=admin,
        json={
            "max_documents": 2,
            "max_storage_bytes": 20,
            "monthly_embedding_tokens": 50,
        },
    )
    assert replaced.status_code == 200, replaced.text
    assert replaced.json()["documents"] == {"current": 1, "limit": 2, "remaining": 1}
    assert replaced.json()["storage_bytes"] == {
        "current": 17,
        "limit": 20,
        "remaining": 3,
    }

    audit = client.get(
        "/api/security/audit",
        headers=admin,
        params={"action": "usage.limits.replace", "limit": 1},
    )
    assert audit.status_code == 200, audit.text
    assert audit.json()["total"] == 1
    event = audit.json()["items"][0]
    assert event["resource_id"] == user["id"]
    assert event["details"]["after"]["monthly_embedding_limit"] == 50
    assert client.get("/api/usage/users/missing", headers=admin).status_code == 404


def test_limit_replacement_is_complete_and_rejects_unknown_fields(
    admin_config: ConfigFile,
) -> None:
    settings = admin_config.load()
    client = TestClient(bootstrap(settings))
    admin = _login(client, "admin", "correct horse battery staple")
    admin_id = client.get("/api/auth/me", headers=admin).json()["id"]

    missing = client.put(
        f"/api/usage/users/{admin_id}/limits",
        headers=admin,
        json={"max_documents": 5, "max_storage_bytes": 100},
    )
    unknown = client.put(
        f"/api/usage/users/{admin_id}/limits",
        headers=admin,
        json={
            "max_documents": None,
            "max_storage_bytes": None,
            "monthly_embedding_tokens": None,
            "chat_tokens": 1,
        },
    )
    reset = client.put(
        f"/api/usage/users/{admin_id}/limits",
        headers=admin,
        json={
            "max_documents": None,
            "max_storage_bytes": None,
            "monthly_embedding_tokens": None,
        },
    )

    assert missing.status_code == 422
    assert unknown.status_code == 422
    assert reset.status_code == 200
    assert reset.json()["documents"]["limit"] == settings.default_max_documents


def _login(client: TestClient, username: str, password: str) -> dict[str, str]:
    response = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


def _insert_hidden_billed_document(database: Path, creator_id: str) -> None:
    document_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    value = b"hidden aggregate!"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO documents (
                id, title, created_by_user_id, original_filename, media_type,
                storage_key, byte_size, content_hash, extracted_characters,
                chunk_count, index_state, index_error, created_at, updated_at
            ) VALUES (?, 'Hidden title', ?, 'hidden.txt', 'text/plain', ?, ?, ?,
                      0, 0, 'failed', 'interrupted', ?, ?)
            """,
            (
                document_id,
                creator_id,
                f"{document_id}/source",
                len(value),
                hashlib.sha256(value).hexdigest(),
                now,
                now,
            ),
        )
