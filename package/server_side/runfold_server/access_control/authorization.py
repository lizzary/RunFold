from __future__ import annotations

import sqlite3
from pathlib import Path

from runfold_server.access_control.capabilities import ROOT_CAPABILITIES, SYSTEM_ADMIN_ROLE_ID
from runfold_server.access_control.models import CurrentAccess
from runfold_server.access_control.repository import AccessControlRepository
from runfold_server.errors import ApiError
from runfold_server.storage.sqlite import connect


class AuthorizationService:
    def __init__(self, database_path: Path, repository: AccessControlRepository) -> None:
        self._database_path = database_path
        self._repository = repository

    def require_capabilities(
        self,
        user_id: str,
        required: frozenset[str],
        *,
        connection: sqlite3.Connection | None = None,
    ) -> CurrentAccess:
        if connection is None:
            with connect(self._database_path) as current_connection:
                return self.require_capabilities(
                    user_id, required, connection=current_connection
                )

        access = self._repository.current_access(connection, user_id)
        if access is None:
            raise ApiError(403, "permission_denied", "Permission denied")
        if SYSTEM_ADMIN_ROLE_ID not in access.role_ids:
            access = CurrentAccess(
                user_id=access.user_id,
                role_ids=access.role_ids,
                capabilities=access.capabilities.difference(ROOT_CAPABILITIES),
            )
        if not required.issubset(access.capabilities):
            raise ApiError(403, "permission_denied", "Permission denied")
        return access
