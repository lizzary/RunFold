from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from runfold_server.bootstrap import bootstrap
from runfold_server.config import load_settings

_RUN_REAL_E2E = os.environ.get("RUNFOLD_REAL_E2E") == "1"


@pytest.mark.skipif(not _RUN_REAL_E2E, reason="set RUNFOLD_REAL_E2E=1")
def test_real_upload_index_search_agent_answer(tmp_path: Path) -> None:
    server_root = Path(__file__).parents[2]
    values = yaml.safe_load(
        (server_root / "config.example.yaml").read_text(encoding="utf-8")
    )
    values["data"]["directory"] = str((tmp_path / "data").resolve())
    values["auth"]["bootstrap_admin"] = {
        "username": "real-e2e-admin",
        "password": "real e2e administrator password",
    }
    document_paths = sorted((server_root / "tests" / "doc").glob("*.docx"))
    values["rag"]["upload_max_bytes"] = max(
        path.stat().st_size for path in document_paths
    )
    values["rag"]["docx_max_uncompressed_bytes"] = max(
        _docx_uncompressed_bytes(path) for path in document_paths
    )
    config_path = tmp_path / "real-e2e.yaml"
    config_path.write_text(
        yaml.safe_dump(values, sort_keys=False),
        encoding="utf-8",
    )
    with TestClient(bootstrap(load_settings(config_path))) as client:
        login = client.post(
            "/api/auth/login",
            json={
                "username": "real-e2e-admin",
                "password": "real e2e administrator password",
            },
        )
        assert login.status_code == 200, login.text
        headers = {"Authorization": f"Bearer {login.json()['token']}"}

        document_ids: dict[str, str] = {}
        for document_path in document_paths:
            uploaded = client.post(
                "/api/rag/documents",
                headers=headers,
                data={"title": f"Real Agent E2E {document_path.stem}"},
                files={
                    "file": (
                        document_path.name,
                        document_path.read_bytes(),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
            )
            assert uploaded.status_code == 201, uploaded.text
            assert uploaded.json()["index_state"] == "ready"
            document_ids[document_path.name] = uploaded.json()["id"]
        document_id = next(
            document_id
            for name, document_id in document_ids.items()
            if "W3C check" in name
        )

        searched = client.post(
            "/api/rag/search",
            headers=headers,
            json={
                "query": "What function is missing?",
                "top_k": 3,
                "document_ids": [document_id],
            },
        )
        assert searched.status_code == 200, searched.text
        assert searched.json()["items"]
        assert "Missing tap function" in searched.json()["items"][0]["text"]

        agent = client.post(
            "/api/agent/runs",
            headers=headers,
            json={
                "input": (
                    "Do not delegate. Call search_knowledge exactly once with query "
                    f"'What function is missing?', top_k 3, and document_ids ['{document_id}']. "
                    "Then state the exact missing function in the final answer."
                )
            },
        )
        assert agent.status_code == 200, agent.text
        assert "missing tap function" in agent.json()["answer"].lower()
        assert agent.json()["reasoning_content"]
        assert agent.json()["thinking_level"] is None
        assert agent.json()["agents_created"] == 0

        usage = client.get("/api/usage/me", headers=headers)
        assert usage.status_code == 200, usage.text
        assert usage.json()["agent_tokens"]["current"] > 0

        audit = client.get(
            "/api/security/audit",
            headers=headers,
            params={"action": "agent.run", "limit": 10},
        )
        assert audit.status_code == 200, audit.text
        assert audit.json()["items"][0]["decision"] == "allowed"

        for uploaded_id in document_ids.values():
            deleted = client.delete(
                f"/api/rag/documents/{uploaded_id}",
                headers=headers,
            )
            assert deleted.status_code == 204, deleted.text


def _docx_uncompressed_bytes(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(item.file_size for item in archive.infolist())
