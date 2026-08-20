from __future__ import annotations

import pytest
from conftest import ConfigFile
from fastapi.testclient import TestClient

from runfold_server.__main__ import main
from runfold_server.bootstrap import bootstrap
from runfold_server.errors import StartupError
from runfold_server.storage.sqlite import connect


def test_bootstrap_initializes_empty_directory_and_reuses_database(
    admin_config: ConfigFile,
) -> None:
    settings = admin_config.load()

    first = bootstrap(settings)
    second = bootstrap(settings)

    assert TestClient(first).get("/health/ready").status_code == 200
    assert TestClient(second).get("/health/ready").status_code == 200
    assert (admin_config.data_dir / "runfold.sqlite3").is_file()


def test_cli_rejects_multiple_workers_before_loading_configuration() -> None:
    assert main(["--workers", "2"]) == 2


def test_empty_user_database_requires_bootstrap_credentials(
    valid_config: ConfigFile,
) -> None:
    settings = valid_config.load()

    with pytest.raises(StartupError) as captured:
        bootstrap(settings)

    assert captured.value.code == "bootstrap_admin_required"


def test_mutable_roles_survive_current_schema_revalidation(
    admin_config: ConfigFile,
) -> None:
    settings = admin_config.load()
    bootstrap(settings)
    database = admin_config.data_dir / "runfold.sqlite3"
    with connect(database) as connection:
        connection.execute(
            "UPDATE roles SET description = 'changed' WHERE name = 'reader'"
        )

    assert TestClient(bootstrap(settings)).get("/health/ready").status_code == 200
