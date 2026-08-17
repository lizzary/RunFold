from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import FastAPI

from runfold_server.config import Settings, load_settings
from runfold_server.http.app import create_app
from runfold_server.storage.sqlite import (
    DataPaths,
    check_database_ready,
    check_local_directories,
    initialize_data_paths,
    initialize_database,
)

_LOGGER = logging.getLogger("runfold_server.bootstrap")


def bootstrap(settings: Settings | None = None) -> FastAPI:
    current_settings = load_settings() if settings is None else settings
    paths = initialize_data_paths(current_settings.data_dir)
    initialize_database(paths.database)
    readiness_check = _readiness_check(paths)
    if not readiness_check():
        raise RuntimeError("Local infrastructure did not become ready")
    return create_app(
        allowed_origins=current_settings.allowed_origins,
        readiness_check=readiness_check,
    )


def _readiness_check(paths: DataPaths) -> Callable[[], bool]:
    def check() -> bool:
        try:
            check_database_ready(paths.database)
            check_local_directories(paths)
        except Exception as error:
            _LOGGER.warning(
                "readiness_check_failed",
                extra={"reason": type(error).__name__},
            )
            return False
        return True

    return check
