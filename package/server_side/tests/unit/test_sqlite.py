from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from runfold_server.errors import StartupError
from runfold_server.storage.sqlite import (
    check_database_ready,
    connect,
    initialize_data_paths,
    initialize_database,
)


def test_data_paths_and_database_initialize_and_reuse(tmp_path: Path) -> None:
    paths = initialize_data_paths(tmp_path / "data")

    initialize_database(paths.database)
    initialize_database(paths.database)

    assert paths.objects.is_dir()
    assert paths.lance.is_dir()
    assert paths.staging.is_dir()
    with connect(paths.database) as connection:
        assert connection.execute("SELECT status FROM service_state").fetchone()[0] == "ready"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 1
    check_database_ready(paths.database)


def test_existing_unknown_schema_is_not_modified(tmp_path: Path) -> None:
    paths = initialize_data_paths(tmp_path / "data")
    with sqlite3.connect(paths.database) as connection:
        connection.execute("CREATE TABLE foreign_table (id INTEGER PRIMARY KEY)")

    with pytest.raises(StartupError) as captured:
        initialize_database(paths.database)

    assert captured.value.code == "incompatible_database_schema"
    with sqlite3.connect(paths.database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    assert tables == {"foreign_table"}


def test_extra_column_fails_exact_schema_assertion(tmp_path: Path) -> None:
    paths = initialize_data_paths(tmp_path / "data")
    initialize_database(paths.database)
    with sqlite3.connect(paths.database) as connection:
        connection.execute("ALTER TABLE service_state ADD COLUMN unexpected TEXT")

    with pytest.raises(StartupError) as captured:
        initialize_database(paths.database)

    assert captured.value.code == "incompatible_database_schema"


def test_changed_fixed_seed_is_rejected(tmp_path: Path) -> None:
    paths = initialize_data_paths(tmp_path / "data")
    initialize_database(paths.database)
    with sqlite3.connect(paths.database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute("UPDATE service_state SET status = 'changed'")

    with pytest.raises(StartupError) as captured:
        initialize_database(paths.database)

    assert captured.value.code == "incompatible_database_seed"


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE capabilities SET description = 'changed' WHERE code = 'identity.user.read'",
        "DELETE FROM role_capabilities WHERE role_id = "
        "'00000000-0000-4000-8000-000000000001' AND "
        "capability_code = 'identity.user.read'",
        "UPDATE roles SET name = 'changed' WHERE id = "
        "'00000000-0000-4000-8000-000000000001'",
    ],
)
def test_changed_identity_fixed_seed_is_rejected(tmp_path: Path, statement: str) -> None:
    paths = initialize_data_paths(tmp_path / "data")
    initialize_database(paths.database)
    with sqlite3.connect(paths.database) as connection:
        connection.execute(statement)

    with pytest.raises(StartupError) as captured:
        initialize_database(paths.database)

    assert captured.value.code == "incompatible_database_seed"


def test_test_suite_has_only_the_production_schema() -> None:
    server_root = Path(__file__).parents[2]

    assert list(server_root.rglob("*.sql")) == [
        server_root / "runfold_server" / "storage" / "schema.sql"
    ]
