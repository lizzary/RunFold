from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from runfold_server.bootstrap import bootstrap
from runfold_server.config import load_settings
from runfold_server.knowledge.lance_index import LanceIndex
from runfold_server.llm.openai_embeddings import EmbeddingBatch, OpenAIEmbeddingsClient

READER_ROLE_ID = "00000000-0000-4000-8000-000000000004"


@pytest.fixture
def search_client(
    admin_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, Path]:
    async def embed(
        self: OpenAIEmbeddingsClient, values: tuple[str, ...]
    ) -> EmbeddingBatch:
        vectors = tuple(_vector(value) for value in values)
        return EmbeddingBatch(vectors=vectors, total_tokens=len(values) * 5)

    monkeypatch.setattr(OpenAIEmbeddingsClient, "embed", embed)
    settings = load_settings(admin_environment)
    return TestClient(bootstrap(settings)), settings.data_dir


def test_search_scope_is_exact_and_never_returns_an_unauthorized_document(
    search_client: tuple[TestClient, Path],
) -> None:
    client, data_dir = search_client
    admin = _login(client, "admin", "correct horse battery staple")
    reader = _create_user(client, admin, "search-reader", READER_ROLE_ID)
    no_acl_reader = _create_user(client, admin, "empty-reader", READER_ROLE_ID)
    no_role = _create_user(client, admin, "acl-without-capability", None)
    reader_headers = _login(client, reader["username"], "user password 123")
    empty_headers = _login(client, no_acl_reader["username"], "user password 123")
    no_role_headers = _login(client, no_role["username"], "user password 123")

    readable = _upload(client, admin, "Readable", "red-public.txt", b"red public facts")
    hidden = _upload(client, admin, "Hidden", "blue-secret.txt", b"blue secret facts")
    _grant_read(client, admin, readable["id"], reader["id"])
    _grant_read(client, admin, readable["id"], no_role["id"])

    omitted = client.post(
        "/api/rag/search",
        headers=reader_headers,
        json={"query": "blue secret facts", "top_k": 8},
    )
    assert omitted.status_code == 200, omitted.text
    assert omitted.json()["items"]
    assert {item["document_id"] for item in omitted.json()["items"]} == {
        readable["id"]
    }
    assert hidden["id"] not in omitted.text
    assert "blue secret facts" not in omitted.text

    explicit = client.post(
        "/api/rag/search",
        headers=reader_headers,
        json={
            "query": "red public facts",
            "top_k": 2,
            "document_ids": [readable["id"]],
        },
    )
    assert explicit.status_code == 200
    assert {item["document_id"] for item in explicit.json()["items"]} == {
        readable["id"]
    }

    for scope in ([], None):
        invalid = client.post(
            "/api/rag/search",
            headers=reader_headers,
            json={"query": "red", "top_k": 2, "document_ids": scope},
        )
        assert invalid.status_code == 422
        assert invalid.json()["code"] == "invalid_document_scope"

    mixed = client.post(
        "/api/rag/search",
        headers=reader_headers,
        json={
            "query": "red",
            "top_k": 2,
            "document_ids": [readable["id"], hidden["id"]],
        },
    )
    assert mixed.status_code == 404
    assert mixed.json()["code"] == "document_not_found"
    assert hidden["id"] not in mixed.text

    empty = client.post(
        "/api/rag/search",
        headers=empty_headers,
        json={"query": "blue secret facts", "top_k": 8},
    )
    assert empty.status_code == 200
    assert empty.json() == {"items": []}
    with sqlite3.connect(data_dir / "runfold.sqlite3") as connection:
        usage = connection.execute(
            """
            SELECT embedding_tokens, search_requests
            FROM usage_monthly
            WHERE user_id = ?
            """,
            (no_acl_reader["id"],),
        ).fetchone()
    assert usage == (0, 1)

    denied = client.post(
        "/api/rag/search",
        headers=no_role_headers,
        json={"query": "red", "top_k": 2},
    )
    assert denied.status_code == 403
    assert denied.json()["code"] == "permission_denied"


def test_search_query_usage_and_summary_audit_exclude_the_query(
    search_client: tuple[TestClient, Path],
) -> None:
    client, data_dir = search_client
    admin = _login(client, "admin", "correct horse battery staple")
    reader = _create_user(client, admin, "usage-reader", READER_ROLE_ID)
    reader_headers = _login(client, reader["username"], "user password 123")
    document = _upload(client, admin, "Reference", "reference.txt", b"green reference")
    _grant_read(client, admin, document["id"], reader["id"])
    query = "green full query must stay private"

    response = client.post(
        "/api/rag/search",
        headers=reader_headers,
        json={"query": query, "top_k": 3},
    )

    assert response.status_code == 200, response.text
    assert response.json()["items"][0]["title"] == "Reference"
    with sqlite3.connect(data_dir / "runfold.sqlite3") as connection:
        usage = connection.execute(
            """
            SELECT embedding_tokens, search_requests
            FROM usage_monthly
            WHERE user_id = ?
            """,
            (reader["id"],),
        ).fetchone()
        event = connection.execute(
            """
            SELECT decision, reason, details_json
            FROM audit_events
            WHERE actor_user_id = ? AND action = 'rag.search'
            ORDER BY id DESC LIMIT 1
            """,
            (reader["id"],),
        ).fetchone()
    assert usage == (5, 1)
    assert event[0:2] == ("allowed", None)
    details = json.loads(event[2])
    assert details["authorized_count"] == 1
    assert details["reference_ids"] == [document["id"]]
    assert query not in event[2]


def test_unknown_document_and_old_hash_results_fail_closed_and_are_audited(
    search_client: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, data_dir = search_client
    admin = _login(client, "admin", "correct horse battery staple")
    reader = _create_user(client, admin, "stale-reader", READER_ROLE_ID)
    reader_headers = _login(client, reader["username"], "user password 123")
    document = _upload(client, admin, "Current", "current.txt", b"current material")
    _grant_read(client, admin, document["id"], reader["id"])
    original_search = LanceIndex.search

    def stale_search(self: LanceIndex, vector, *, document_ids, top_k):
        hits = original_search(
            self, vector, document_ids=document_ids, top_k=top_k
        )
        assert hits
        return (replace(hits[0], content_hash="0" * 64),)

    monkeypatch.setattr(LanceIndex, "search", stale_search)
    response = client.post(
        "/api/rag/search",
        headers=reader_headers,
        json={"query": "current material", "top_k": 3},
    )

    assert response.status_code == 503
    assert response.json()["code"] == "unsafe_index_result"
    assert "current material" not in response.text
    with sqlite3.connect(data_dir / "runfold.sqlite3") as connection:
        event = connection.execute(
            """
            SELECT decision, reason, details_json
            FROM audit_events
            WHERE actor_user_id = ? AND action = 'rag.search'
            ORDER BY id DESC LIMIT 1
            """,
            (reader["id"],),
        ).fetchone()
    assert event[0:2] == ("denied", "unsafe_index_result")
    assert "current material" not in event[2]

    unknown_id = "00000000-0000-4000-8000-000000000999"

    def unknown_search(self: LanceIndex, vector, *, document_ids, top_k):
        hits = original_search(
            self, vector, document_ids=document_ids, top_k=top_k
        )
        assert hits
        return (replace(hits[0], document_id=unknown_id),)

    monkeypatch.setattr(LanceIndex, "search", unknown_search)
    unknown = client.post(
        "/api/rag/search",
        headers=reader_headers,
        json={"query": "current material", "top_k": 3},
    )
    assert unknown.status_code == 503
    assert unknown.json()["code"] == "unsafe_index_result"
    assert unknown_id not in unknown.text
    with sqlite3.connect(data_dir / "runfold.sqlite3") as connection:
        last_details = connection.execute(
            """
            SELECT details_json
            FROM audit_events
            WHERE actor_user_id = ? AND action = 'rag.search'
            ORDER BY id DESC LIMIT 1
            """,
            (reader["id"],),
        ).fetchone()[0]
    assert json.loads(last_details)["suspect_ids"] == [unknown_id]
    assert "current material" not in last_details


def test_acl_revoked_during_lance_search_prevents_any_result(
    search_client: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, data_dir = search_client
    admin = _login(client, "admin", "correct horse battery staple")
    reader = _create_user(client, admin, "revoked-reader", READER_ROLE_ID)
    reader_headers = _login(client, reader["username"], "user password 123")
    document = _upload(client, admin, "Revocable", "revocable.txt", b"revocable facts")
    _grant_read(client, admin, document["id"], reader["id"])
    original_search = LanceIndex.search

    def revoke_after_search(self: LanceIndex, vector, *, document_ids, top_k):
        hits = original_search(
            self, vector, document_ids=document_ids, top_k=top_k
        )
        with sqlite3.connect(data_dir / "runfold.sqlite3") as connection:
            connection.execute(
                "DELETE FROM document_acl WHERE document_id = ? AND user_id = ?",
                (document["id"], reader["id"]),
            )
        return hits

    monkeypatch.setattr(LanceIndex, "search", revoke_after_search)
    response = client.post(
        "/api/rag/search",
        headers=reader_headers,
        json={"query": "revocable facts", "top_k": 3},
    )

    assert response.status_code == 503
    assert response.json()["code"] == "unsafe_index_result"
    assert "revocable facts" not in response.text


def test_search_embedding_quota_is_checked_before_external_work(
    search_client: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, data_dir = search_client
    admin = _login(client, "admin", "correct horse battery staple")
    reader = _create_user(client, admin, "quota-reader", READER_ROLE_ID)
    reader_headers = _login(client, reader["username"], "user password 123")
    document = _upload(client, admin, "Quota", "quota.txt", b"quota facts")
    _grant_read(client, admin, document["id"], reader["id"])
    with sqlite3.connect(data_dir / "runfold.sqlite3") as connection:
        connection.execute(
            """
            INSERT INTO usage_monthly (
                user_id, month_utc, embedding_tokens, search_requests, uploads, updated_at
            ) VALUES (?, strftime('%Y-%m', 'now'), 1000000, 0, 0, 'now')
            """,
            (reader["id"],),
        )

    called = False

    async def forbidden_embed(self: OpenAIEmbeddingsClient, values: tuple[str, ...]):
        nonlocal called
        called = True
        raise AssertionError("embedding must not be called")

    monkeypatch.setattr(OpenAIEmbeddingsClient, "embed", forbidden_embed)
    response = client.post(
        "/api/rag/search",
        headers=reader_headers,
        json={"query": "quota facts", "top_k": 3},
    )

    assert response.status_code == 429
    assert response.json()["code"] == "quota_exceeded"
    assert not called
    with sqlite3.connect(data_dir / "runfold.sqlite3") as connection:
        event = connection.execute(
            """
            SELECT decision, reason, details_json
            FROM audit_events
            WHERE actor_user_id = ? AND action = 'rag.search'
            ORDER BY id DESC LIMIT 1
            """,
            (reader["id"],),
        ).fetchone()
    assert event[0:2] == ("denied", "quota_exceeded")
    assert json.loads(event[2])["quota"] == "embedding_tokens"


def _vector(value: str) -> tuple[float, ...]:
    lowered = value.lower()
    if "blue" in lowered:
        return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if "red" in lowered:
        return (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _login(client: TestClient, username: str, password: str) -> dict[str, str]:
    response = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


def _create_user(
    client: TestClient,
    admin: dict[str, str],
    username: str,
    role_id: str | None,
) -> dict[str, str]:
    created = client.post(
        "/api/access/users",
        headers=admin,
        json={
            "username": username,
            "display_name": username,
            "password": "user password 123",
        },
    )
    assert created.status_code == 201, created.text
    user = created.json()
    assert client.put(
        f"/api/access/users/{user['id']}/roles",
        headers=admin,
        json={"role_ids": [] if role_id is None else [role_id]},
    ).status_code == 200
    return user


def _upload(
    client: TestClient,
    headers: dict[str, str],
    title: str,
    filename: str,
    content: bytes,
) -> dict[str, object]:
    response = client.post(
        "/api/rag/documents",
        headers=headers,
        data={"title": title},
        files={"file": (filename, content)},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _grant_read(
    client: TestClient,
    admin: dict[str, str],
    document_id: str,
    user_id: str,
) -> None:
    current = client.get(f"/api/rag/documents/{document_id}/acl", headers=admin)
    assert current.status_code == 200
    grants = current.json()["grants"]
    grants.append({"user_id": user_id, "role_id": None, "access_level": 10})
    response = client.put(
        f"/api/rag/documents/{document_id}/acl",
        headers=admin,
        json={"grants": grants},
    )
    assert response.status_code == 200, response.text
