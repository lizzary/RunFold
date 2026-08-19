from __future__ import annotations

import sqlite3

import lancedb
import pytest
from fastapi.testclient import TestClient

from runfold_server.__main__ import main
from runfold_server.bootstrap import bootstrap
from runfold_server.config import load_settings
from runfold_server.knowledge.lance_index import LanceIndex
from runfold_server.llm.openai_embeddings import EmbeddingBatch, OpenAIEmbeddingsClient


def test_stopped_service_rebuild_changes_dimensions_bills_actor_and_compacts(
    admin_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def embed(
        self: OpenAIEmbeddingsClient, values: tuple[str, ...]
    ) -> EmbeddingBatch:
        return EmbeddingBatch(
            vectors=tuple((0.25,) * self._dimensions for _ in values),
            total_tokens=len(values) * 5,
        )

    monkeypatch.setattr(OpenAIEmbeddingsClient, "embed", embed)
    settings = load_settings(admin_environment)
    with TestClient(bootstrap(settings)) as client:
        headers = _login(client)
        document_id = _upload(client, headers)

    changed_environment = dict(admin_environment)
    changed_environment["RUNFOLD_EMBEDDING_DIMENSIONS"] = "4"
    assert main(["rebuild-index", "--actor", "admin"], changed_environment) == 0

    table = lancedb.connect(settings.data_dir / "lance").open_table("chunks")
    assert table.schema.field("vector").type.list_size == 4
    assert table.count_rows(f"document_id = '{document_id}'") > 0
    with sqlite3.connect(settings.data_dir / "runfold.sqlite3") as connection:
        state = connection.execute(
            "SELECT index_state FROM documents WHERE id = ?", (document_id,)
        ).fetchone()[0]
        actor_id = connection.execute(
            "SELECT id FROM users WHERE username = 'admin'"
        ).fetchone()[0]
        billed = connection.execute(
            "SELECT embedding_tokens FROM usage_monthly WHERE user_id = ?",
            (actor_id,),
        ).fetchone()[0]
        audit_count = connection.execute(
            """
            SELECT COUNT(*) FROM audit_events
            WHERE actor_user_id = ? AND action = 'rag.index.rebuild'
            """,
            (actor_id,),
        ).fetchone()[0]
    assert state == "ready"
    assert billed >= 10
    assert audit_count == 1
    assert main(["compact-index"], changed_environment) == 0
    assert main(["rebuild-index", "--actor", "missing"], changed_environment) == 2


def test_rebuild_interruption_after_settings_update_converges_without_embeddings(
    admin_environment: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def embed(
        self: OpenAIEmbeddingsClient, values: tuple[str, ...]
    ) -> EmbeddingBatch:
        return EmbeddingBatch(
            vectors=tuple((0.5,) * self._dimensions for _ in values),
            total_tokens=len(values),
        )

    monkeypatch.setattr(OpenAIEmbeddingsClient, "embed", embed)
    settings = load_settings(admin_environment)
    with TestClient(bootstrap(settings)) as client:
        document_id = _upload(client, _login(client))

    changed_environment = dict(admin_environment)
    changed_environment["RUNFOLD_EMBEDDING_DIMENSIONS"] = "4"
    original_recreate = LanceIndex.recreate

    def interrupt(self: LanceIndex) -> None:
        raise RuntimeError("injected interruption")

    monkeypatch.setattr(LanceIndex, "recreate", interrupt)
    assert main(["rebuild-index", "--actor", "admin"], changed_environment) == 1
    database = settings.data_dir / "runfold.sqlite3"
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT index_state FROM documents WHERE id = ?", (document_id,)
        ).fetchone()[0] == "indexing"
        assert connection.execute(
            "SELECT dimensions FROM rag_index_settings WHERE singleton = 1"
        ).fetchone()[0] == 4

    monkeypatch.setattr(LanceIndex, "recreate", original_recreate)

    async def forbidden_embed(
        self: OpenAIEmbeddingsClient, values: tuple[str, ...]
    ) -> EmbeddingBatch:
        raise AssertionError("startup recovery must not call embeddings")

    monkeypatch.setattr(OpenAIEmbeddingsClient, "embed", forbidden_embed)
    bootstrap(load_settings(changed_environment))

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT index_state, index_error FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone() == ("failed", "interrupted")
    vector_type = lancedb.connect(settings.data_dir / "lance").open_table(
        "chunks"
    ).schema.field("vector").type
    assert vector_type.list_size == 4


def _login(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "correct horse battery staple"},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['token']}"}


def _upload(client: TestClient, headers: dict[str, str]) -> str:
    response = client.post(
        "/api/rag/documents",
        headers=headers,
        data={"title": "Maintenance source"},
        files={"file": ("maintenance.txt", b"maintenance content")},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])
