from __future__ import annotations

import sqlite3

import pytest
from conftest import ConfigFile
from fastapi.testclient import TestClient

from runfold_server.bootstrap import bootstrap
from runfold_server.knowledge.lance_index import LanceIndex
from runfold_server.knowledge.object_store import ObjectStore
from runfold_server.llm.openai_embeddings import EmbeddingBatch, OpenAIEmbeddingsClient


def test_replace_reindex_and_delete_faults_restart_to_safe_terminal_states(
    admin_config: ConfigFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def embed(
        self: OpenAIEmbeddingsClient, values: tuple[str, ...]
    ) -> EmbeddingBatch:
        return EmbeddingBatch(
            vectors=tuple((0.1,) * self._dimensions for _ in values),
            total_tokens=len(values),
        )

    monkeypatch.setattr(OpenAIEmbeddingsClient, "embed", embed)
    settings = admin_config.load()
    original_replace = LanceIndex.replace_document
    original_delete_object = ObjectStore.delete_document
    with TestClient(bootstrap(settings)) as client:
        headers = _login(client)
        replace_id = _upload(client, headers, "replace.txt")
        reindex_id = _upload(client, headers, "reindex.txt")
        delete_id = _upload(client, headers, "delete.txt")

        async def fail_index(
            self: LanceIndex,
            document_id: str,
            content_hash: str,
            chunks,
            vectors,
        ) -> None:
            raise RuntimeError("injected index failure")

        monkeypatch.setattr(LanceIndex, "replace_document", fail_index)
        replaced = client.put(
            f"/api/rag/documents/{replace_id}/text",
            headers=headers,
            json={"text": "replacement content"},
        )
        reindexed = client.post(
            f"/api/rag/documents/{reindex_id}/reindex", headers=headers
        )
        assert replaced.status_code == 503
        assert reindexed.status_code == 503
        monkeypatch.setattr(LanceIndex, "replace_document", original_replace)

        def fail_object_delete(self: ObjectStore, document_id: str) -> None:
            raise OSError("injected object delete failure")

        monkeypatch.setattr(ObjectStore, "delete_document", fail_object_delete)
        deleted = client.delete(f"/api/rag/documents/{delete_id}", headers=headers)
        assert deleted.status_code == 503
        assert deleted.json()["code"] == "document_delete_incomplete"
        monkeypatch.setattr(ObjectStore, "delete_document", original_delete_object)

    bootstrap(settings)

    database = settings.data_dir / "runfold.sqlite3"
    with sqlite3.connect(database) as connection:
        states = dict(
            connection.execute(
                "SELECT id, index_state FROM documents WHERE id IN (?, ?, ?)",
                (replace_id, reindex_id, delete_id),
            ).fetchall()
        )
    assert states == {replace_id: "failed", reindex_id: "failed"}
    assert not (settings.data_dir / "objects" / replace_id / "extracted.txt").exists()
    assert not (settings.data_dir / "objects" / reindex_id / "extracted.txt").exists()
    assert not (settings.data_dir / "objects" / delete_id).exists()


def _login(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "correct horse battery staple"},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['token']}"}


def _upload(client: TestClient, headers: dict[str, str], filename: str) -> str:
    response = client.post(
        "/api/rag/documents",
        headers=headers,
        data={"title": filename},
        files={"file": (filename, f"content for {filename}".encode())},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])
