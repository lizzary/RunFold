from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from runfold_server.__main__ import main
from runfold_server.bootstrap import bootstrap
from runfold_server.config import load_settings
from runfold_server.errors import StartupError
from runfold_server.storage.sqlite import connect


def test_bootstrap_initializes_empty_directory_and_reuses_database(
    admin_environment: dict[str, str],
) -> None:
    settings = load_settings(admin_environment)

    first = bootstrap(settings)
    second = bootstrap(settings)

    assert TestClient(first).get("/health/ready").status_code == 200
    assert TestClient(second).get("/health/ready").status_code == 200
    assert (Path(admin_environment["RUNFOLD_DATA_DIR"]) / "runfold.sqlite3").is_file()


def test_cli_rejects_multiple_workers_before_loading_configuration() -> None:
    assert main(["--workers", "2"], {}) == 2
    assert main([], {"WEB_CONCURRENCY": "4"}) == 2


def test_empty_user_database_requires_bootstrap_credentials(
    valid_environment: dict[str, str],
) -> None:
    settings = load_settings(valid_environment)

    with pytest.raises(StartupError) as captured:
        bootstrap(settings)

    assert captured.value.code == "bootstrap_admin_required"


def test_mutable_roles_survive_current_schema_revalidation(
    admin_environment: dict[str, str],
) -> None:
    settings = load_settings(admin_environment)
    bootstrap(settings)
    database = Path(admin_environment["RUNFOLD_DATA_DIR"]) / "runfold.sqlite3"
    with connect(database) as connection:
        connection.execute(
            "UPDATE roles SET description = 'changed' WHERE name = 'reader'"
        )

    assert TestClient(bootstrap(settings)).get("/health/ready").status_code == 200
