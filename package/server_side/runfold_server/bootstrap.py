from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import FastAPI

from runfold_server.access_control.audit import AuditRepository
from runfold_server.access_control.authorization import AuthorizationService
from runfold_server.access_control.repository import AccessControlRepository
from runfold_server.access_control.service import AccessControlService
from runfold_server.config import Settings, load_settings
from runfold_server.http.app import create_app
from runfold_server.identity.passwords import Argon2PasswordHasher
from runfold_server.identity.repository import IdentityRepository
from runfold_server.identity.service import IdentityService
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
    audit_repository = AuditRepository()
    identity_repository = IdentityRepository()
    access_repository = AccessControlRepository()
    password_hasher = Argon2PasswordHasher()
    identity_service = IdentityService(
        database_path=paths.database,
        repository=identity_repository,
        password_hasher=password_hasher,
        audit=audit_repository,
        session_ttl_seconds=current_settings.session_ttl_seconds,
    )
    authorization_service = AuthorizationService(paths.database, access_repository)
    access_control_service = AccessControlService(
        database_path=paths.database,
        identity=identity_service,
        identity_repository=identity_repository,
        repository=access_repository,
        authorization=authorization_service,
        audit=audit_repository,
    )
    access_control_service.ensure_administrator_exists(
        current_settings.bootstrap_admin_username,
        current_settings.bootstrap_admin_password,
    )
    if current_settings.bootstrap_admin_password is not None:
        _LOGGER.warning("bootstrap_admin_credentials_should_be_removed")
    readiness_check = _readiness_check(paths)
    if not readiness_check():
        raise RuntimeError("Local infrastructure did not become ready")
    return create_app(
        allowed_origins=current_settings.allowed_origins,
        readiness_check=readiness_check,
        identity_service=identity_service,
        access_control_service=access_control_service,
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
