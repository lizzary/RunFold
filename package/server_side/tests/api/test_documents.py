from __future__ import annotations

import sqlite3
from pathlib import Path

import lancedb
import pytest
from conftest import ConfigFile
from fastapi.testclient import TestClient

from runfold_server.bootstrap import bootstrap
from runfold_server.knowledge.lance_index import LanceIndex
from runfold_server.llm.openai_embeddings import EmbeddingBatch, OpenAIEmbeddingsClient

CONTRIBUTOR_ROLE_ID = "00000000-0000-4000-8000-000000000003"
READER_ROLE_ID = "00000000-0000-4000-8000-000000000004"


@pytest.fixture
def document_client(
    admin_config: ConfigFile, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, Path]:
    async def embed(
        self: OpenAIEmbeddingsClient, values: tuple[str, ...]
    ) -> EmbeddingBatch:
        vectors = tuple(
            tuple(float(index + 1) / 10 for index in range(8)) for _ in values
        )
        return EmbeddingBatch(vectors=vectors, total_tokens=len(values) * 3)

    monkeypatch.setattr(OpenAIEmbeddingsClient, "embed", embed)
    settings = admin_config.load()
    return TestClient(bootstrap(settings)), settings.data_dir


def test_complete_document_lifecycle_replaces_old_objects_and_vectors(
    document_client: tuple[TestClient, Path],
) -> None:
    client, data_dir = document_client
    headers = _login(client, "admin", "correct horse battery staple")

    uploaded = client.post(
        "/api/rag/documents",
        headers=headers,
        data={"title": "Handbook"},
        files={"file": ("../../handbook.txt", b"first paragraph\n\nsecond paragraph")},
    )

    assert uploaded.status_code == 201, uploaded.text
    document = uploaded.json()
    document_id = document["id"]
    old_hash = document["content_hash"]
    assert document["original_filename"] == "handbook.txt"
    assert document["index_state"] == "ready"
    assert (data_dir / "objects" / document_id / "source").read_bytes().startswith(b"first")
    assert client.get(f"/api/rag/documents/{document_id}/text", headers=headers).json()[
        "text"
    ].startswith("first")

    changed = client.put(
        f"/api/rag/documents/{document_id}/text",
        headers=headers,
        json={"text": "entirely new material"},
    )
    assert changed.status_code == 200, changed.text
    assert changed.json()["content_hash"] != old_hash
    assert client.get(
        f"/api/rag/documents/{document_id}/content", headers=headers
    ).content == b"entirely new material"
    rows = lancedb.connect(data_dir / "lance").open_table("chunks").to_arrow().to_pylist()
    selected = [row for row in rows if row["document_id"] == document_id]
    assert selected
    assert {row["content_hash"] for row in selected} == {changed.json()["content_hash"]}
    assert all("first paragraph" not in row["text"] for row in selected)

    reindexed = client.post(
        f"/api/rag/documents/{document_id}/reindex", headers=headers
    )
    assert reindexed.status_code == 200
    deleted = client.delete(f"/api/rag/documents/{document_id}", headers=headers)
    assert deleted.status_code == 204
    assert client.get(f"/api/rag/documents/{document_id}", headers=headers).status_code == 404
    assert not (data_dir / "objects" / document_id).exists()
    assert (
        lancedb.connect(data_dir / "lance")
        .open_table("chunks")
        .count_rows(f"document_id = '{document_id}'")
        == 0
    )


def test_capability_and_acl_are_both_required_and_revocation_is_immediate(
    document_client: tuple[TestClient, Path],
) -> None:
    client, _ = document_client
    admin = _login(client, "admin", "correct horse battery staple")
    contributor = _create_user(client, admin, "contributor-user", CONTRIBUTOR_ROLE_ID)
    reader = _create_user(client, admin, "reader-user", READER_ROLE_ID)
    no_role = _create_user(client, admin, "no-role-user", None)
    contributor_headers = _login(client, contributor["username"], "user password 123")
    reader_headers = _login(client, reader["username"], "user password 123")
    no_role_headers = _login(client, no_role["username"], "user password 123")

    uploaded = client.post(
        "/api/rag/documents",
        headers=contributor_headers,
        data={"title": "Private"},
        files={"file": ("private.md", b"private words")},
    )
    assert uploaded.status_code == 201
    document_id = uploaded.json()["id"]

    assert (
        client.get(f"/api/rag/documents/{document_id}", headers=reader_headers).status_code
        == 404
    )
    assert client.put(
        f"/api/rag/documents/{document_id}/acl",
        headers=admin,
        json={
            "grants": [
                {"user_id": contributor["id"], "access_level": 30},
                {"user_id": reader["id"], "access_level": 10},
                {"user_id": no_role["id"], "access_level": 30},
            ]
        },
    ).status_code == 200

    assert (
        client.get(f"/api/rag/documents/{document_id}", headers=reader_headers).status_code
        == 200
    )
    assert client.patch(
        f"/api/rag/documents/{document_id}",
        headers=reader_headers,
        json={"title": "not allowed"},
    ).status_code == 403
    assert (
        client.get(f"/api/rag/documents/{document_id}", headers=no_role_headers).status_code
        == 403
    )

    revoked = client.put(
        f"/api/rag/documents/{document_id}/acl",
        headers=admin,
        json={"grants": [{"user_id": contributor["id"], "access_level": 30}]},
    )
    assert revoked.status_code == 200
    assert (
        client.get(f"/api/rag/documents/{document_id}", headers=reader_headers).status_code
        == 404
    )
    page = client.get("/api/rag/documents", headers=reader_headers).json()
    assert page["items"] == [] and page["total"] == 0


def test_invalid_file_and_embedding_quota_fail_without_creating_document(
    document_client: tuple[TestClient, Path],
) -> None:
    client, data_dir = document_client
    headers = _login(client, "admin", "correct horse battery staple")
    database = data_dir / "runfold.sqlite3"

    invalid = client.post(
        "/api/rag/documents",
        headers=headers,
        data={"title": "Fake"},
        files={"file": ("fake.pdf", b"not a pdf")},
    )
    assert invalid.status_code == 422

    with sqlite3.connect(database) as connection:
        user_id = connection.execute("SELECT id FROM users WHERE username = 'admin'").fetchone()[0]
        connection.execute(
            """
            INSERT INTO usage_monthly (user_id, month_utc, embedding_tokens, uploads, updated_at)
            VALUES (?, strftime('%Y-%m', 'now'), 1000000, 0, 'now')
            """,
            (user_id,),
        )
    denied = client.post(
        "/api/rag/documents",
        headers=headers,
        data={"title": "Denied"},
        files={"file": ("denied.txt", b"valid text")},
    )
    assert denied.status_code == 429
    assert denied.json()["code"] == "quota_exceeded"
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    assert list((data_dir / "staging").iterdir()) == []


def test_index_failure_converges_to_failed_and_never_exposes_derived_text(
    document_client: tuple[TestClient, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, data_dir = document_client
    headers = _login(client, "admin", "correct horse battery staple")

    async def fail_index(
        self: LanceIndex,
        document_id: str,
        content_hash: str,
        chunks,
        vectors,
    ) -> None:
        raise RuntimeError("injected index failure")

    monkeypatch.setattr(LanceIndex, "replace_document", fail_index)
    failed = client.post(
        "/api/rag/documents",
        headers=headers,
        data={"title": "Will fail"},
        files={"file": ("failed.txt", b"source remains available")},
    )
    assert failed.status_code == 503
    assert "source remains" not in failed.text

    with sqlite3.connect(data_dir / "runfold.sqlite3") as connection:
        document_id, state, error_code = connection.execute(
            "SELECT id, index_state, index_error FROM documents"
        ).fetchone()
        tokens = connection.execute(
            "SELECT SUM(embedding_tokens) FROM usage_monthly"
        ).fetchone()[0]
    assert (state, error_code) == ("failed", "indexing_failed")
    assert tokens == 3
    assert client.get(
        f"/api/rag/documents/{document_id}/content", headers=headers
    ).status_code == 200
    assert client.get(
        f"/api/rag/documents/{document_id}/text", headers=headers
    ).status_code == 404
    assert not (data_dir / "objects" / document_id / "extracted.txt").exists()


def _login(client: TestClient, username: str, password: str) -> dict[str, str]:
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
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
    assert created.status_code == 201
    user = created.json()
    roles = [] if role_id is None else [role_id]
    assert client.put(
        f"/api/access/users/{user['id']}/roles",
        headers=admin,
        json={"role_ids": roles},
    ).status_code == 200
    return user
