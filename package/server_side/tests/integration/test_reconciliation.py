from __future__ import annotations

import hashlib
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

import lancedb
import pyarrow as pa
import pytest

from runfold_server.bootstrap import bootstrap
from runfold_server.config import load_settings
from runfold_server.errors import StartupError


def test_restart_converges_interrupted_indexing_and_deleting_without_embeddings(
    admin_environment: dict[str, str],
) -> None:
    settings = load_settings(admin_environment)
    bootstrap(settings)
    database = settings.data_dir / "runfold.sqlite3"
    with sqlite3.connect(database) as connection:
        creator_id = connection.execute(
            "SELECT id FROM users WHERE username = 'admin'"
        ).fetchone()[0]

    indexing_id = _insert_document(database, creator_id, "indexing")
    deleting_id = _insert_document(database, creator_id, "deleting")
    for document_id, value in ((indexing_id, b"interrupted"), (deleting_id, b"delete me")):
        directory = settings.data_dir / "objects" / document_id
        directory.mkdir()
        (directory / "source").write_bytes(value)
        (directory / "extracted.txt").write_text("untrusted", encoding="utf-8")
        _add_lance_row(settings.data_dir / "lance", document_id)
    orphan_stage = settings.data_dir / "staging" / str(uuid.uuid4())
    orphan_stage.mkdir()
    (orphan_stage / "partial").write_bytes(b"partial")

    bootstrap(settings)

    with sqlite3.connect(database) as connection:
        failed = connection.execute(
            "SELECT index_state, index_error, chunk_count FROM documents WHERE id = ?",
            (indexing_id,),
        ).fetchone()
        deleted = connection.execute(
            "SELECT 1 FROM documents WHERE id = ?", (deleting_id,)
        ).fetchone()
    assert failed == ("failed", "interrupted", 0)
    assert deleted is None
    assert (settings.data_dir / "objects" / indexing_id / "source").is_file()
    assert not (settings.data_dir / "objects" / indexing_id / "extracted.txt").exists()
    assert not (settings.data_dir / "objects" / deleting_id).exists()
    assert list((settings.data_dir / "staging").iterdir()) == []
    table = lancedb.connect(settings.data_dir / "lance").open_table("chunks")
    assert table.count_rows(f"document_id = '{indexing_id}'") == 0
    assert table.count_rows(f"document_id = '{deleting_id}'") == 0


def test_index_configuration_change_rebuilds_only_when_no_ready_documents(
    admin_environment: dict[str, str],
) -> None:
    settings = load_settings(admin_environment)
    bootstrap(settings)
    changed_environment = dict(admin_environment)
    changed_environment["RUNFOLD_EMBEDDING_DIMENSIONS"] = "4"
    changed = load_settings(changed_environment)

    bootstrap(changed)
    vector_type = lancedb.connect(changed.data_dir / "lance").open_table("chunks").schema.field(
        "vector"
    ).type
    assert vector_type.list_size == 4

    database = changed.data_dir / "runfold.sqlite3"
    with sqlite3.connect(database) as connection:
        creator_id = connection.execute(
            "SELECT id FROM users WHERE username = 'admin'"
        ).fetchone()[0]
    document_id = _insert_document(database, creator_id, "ready")
    directory = changed.data_dir / "objects" / document_id
    directory.mkdir()
    (directory / "source").write_bytes(b"ready")
    (directory / "extracted.txt").write_text("ready", encoding="utf-8")

    original_environment = dict(admin_environment)
    original = load_settings(original_environment)
    with pytest.raises(StartupError) as error:
        bootstrap(original)
    assert error.value.code == "incompatible_rag_index"


def _insert_document(database: Path, creator_id: str, state: str) -> str:
    document_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    value = state.encode()
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO documents (
                id, title, created_by_user_id, original_filename, media_type,
                storage_key, byte_size, content_hash, extracted_characters,
                chunk_count, index_state, index_error, created_at, updated_at
            ) VALUES (?, ?, ?, 'file.txt', 'text/plain', ?, ?, ?, 9, 1, ?, NULL, ?, ?)
            """,
            (
                document_id,
                state,
                creator_id,
                f"{document_id}/source",
                len(value),
                hashlib.sha256(value).hexdigest(),
                state,
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT INTO document_acl (
                document_id, user_id, role_id, access_level, granted_by_user_id, created_at
            ) VALUES (?, ?, NULL, 30, ?, ?)
            """,
            (document_id, creator_id, creator_id, now),
        )
    return document_id


def _add_lance_row(path: Path, document_id: str) -> None:
    table = lancedb.connect(path).open_table("chunks")
    row = pa.Table.from_pylist(
        [
            {
                "document_id": document_id,
                "chunk_id": "chunk",
                "ordinal": 0,
                "content_hash": "stale",
                "text": "untrusted",
                "vector": [0.0] * 8,
            }
        ],
        schema=table.schema,
    )
    table.add(row)
