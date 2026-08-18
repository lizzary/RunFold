from __future__ import annotations

import re
import sqlite3
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path

from runfold_server.access_control.audit import AuditRepository
from runfold_server.access_control.authorization import AuthorizationService
from runfold_server.access_control.capabilities import (
    ALL_CAPABILITIES,
    IDENTITY_ROLE_MANAGE,
    IDENTITY_ROLE_READ,
    IDENTITY_USER_MANAGE,
    IDENTITY_USER_READ,
    ROOT_CAPABILITIES,
    SYSTEM_ADMIN_ROLE_ID,
)
from runfold_server.access_control.models import Capability, Role
from runfold_server.access_control.repository import AccessControlRepository
from runfold_server.errors import ApiError, StartupError
from runfold_server.identity.models import PreparedUser, User, VerifiedIdentity
from runfold_server.identity.repository import IdentityRepository
from runfold_server.identity.service import IdentityService
from runfold_server.storage.sqlite import connect

_ROLE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class AccessControlService:
    def __init__(
        self,
        *,
        database_path: Path,
        identity: IdentityService,
        identity_repository: IdentityRepository,
        repository: AccessControlRepository,
        authorization: AuthorizationService,
        audit: AuditRepository,
    ) -> None:
        self._database_path = database_path
        self._identity = identity
        self._identity_repository = identity_repository
        self._repository = repository
        self._authorization = authorization
        self._audit = audit

    def bootstrap_administrator(self, username: str, password: str) -> None:
        prepared = self._identity.prepare_user(username, username, password)
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._identity_repository.count_users(connection) != 0:
                return
            self._insert_user(connection, prepared)
            self._repository.replace_user_roles(
                connection,
                user_id=prepared.id,
                role_ids=(SYSTEM_ADMIN_ROLE_ID,),
                assigned_by_user_id=prepared.id,
                now=prepared.created_at,
            )
            self._audit.record(
                connection,
                actor_user_id=prepared.id,
                action="identity.user.bootstrap",
                decision="allowed",
                resource_type="user",
                resource_id=prepared.id,
                reason=None,
                request_id="bootstrap",
                details={"role_ids": [SYSTEM_ADMIN_ROLE_ID]},
                now=prepared.created_at,
            )

    def ensure_administrator_exists(
        self, bootstrap_username: str | None, bootstrap_password: str | None
    ) -> None:
        with connect(self._database_path) as connection:
            has_users = self._identity_repository.count_users(connection) > 0
        if has_users:
            return
        if bootstrap_username is None or bootstrap_password is None:
            raise StartupError(
                "bootstrap_admin_required",
                "Bootstrap administrator credentials are required for an empty user database",
            )
        try:
            self.bootstrap_administrator(bootstrap_username, bootstrap_password)
        except ApiError as error:
            raise StartupError("invalid_bootstrap_admin", error.message) from error

    def list_users(
        self, actor: VerifiedIdentity, *, limit: int, offset: int
    ) -> tuple[tuple[User, ...], int]:
        with connect(self._database_path) as connection:
            connection.execute("BEGIN")
            self._require_actor(connection, actor, frozenset({IDENTITY_USER_READ}))
            return (
                self._identity_repository.list_users(
                    connection, limit=limit, offset=offset
                ),
                self._identity_repository.count_users(connection),
            )

    def get_user(self, actor: VerifiedIdentity, user_id: str) -> User:
        with connect(self._database_path) as connection:
            connection.execute("BEGIN")
            self._require_actor(connection, actor, frozenset({IDENTITY_USER_READ}))
            user = self._identity_repository.get_user(connection, user_id)
            if user is None:
                raise ApiError(404, "user_not_found", "User not found")
            return user

    def create_user(
        self,
        actor: VerifiedIdentity,
        *,
        username: str,
        display_name: str,
        password: str,
    ) -> User:
        prepared = self._identity.prepare_user(username, display_name, password)
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._require_actor(
                connection, actor, frozenset({IDENTITY_USER_MANAGE})
            )
            try:
                self._insert_user(connection, prepared)
            except sqlite3.IntegrityError as error:
                raise ApiError(409, "username_exists", "Username already exists") from error
            self._audit.record(
                connection,
                actor_user_id=current.user_id,
                action="identity.user.create",
                decision="allowed",
                resource_type="user",
                resource_id=prepared.id,
                reason=None,
                request_id=current.context.request_id,
                details={"username": prepared.username},
                now=prepared.created_at,
            )
            user = self._identity_repository.get_user(connection, prepared.id)
            assert user is not None
            return user

    def update_user(
        self,
        actor: VerifiedIdentity,
        user_id: str,
        *,
        display_name: str | None,
        status: str | None,
    ) -> User:
        if display_name is None and status is None:
            raise ApiError(422, "invalid_request", "At least one user field is required")
        normalized_display_name = (
            None if display_name is None else self._identity.normalize_display_name(display_name)
        )
        if status is not None and status not in {"active", "disabled"}:
            raise ApiError(422, "invalid_user_status", "User status is invalid")
        now = _now_text()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._require_actor(
                connection, actor, frozenset({IDENTITY_USER_MANAGE})
            )
            before = self._identity_repository.get_user(connection, user_id)
            if before is None:
                raise ApiError(404, "user_not_found", "User not found")
            self._identity_repository.update_user(
                connection,
                user_id=user_id,
                display_name=normalized_display_name or before.display_name,
                status=status or before.status,
                now=now,
            )
            revoked_count = 0
            if before.status == "active" and status == "disabled":
                revoked_count = self._identity_repository.revoke_all_sessions(
                    connection, user_id=user_id, now=now
                )
            self._ensure_active_system_admin(connection)
            after = self._identity_repository.get_user(connection, user_id)
            assert after is not None
            self._audit.record(
                connection,
                actor_user_id=current.user_id,
                action="identity.user.update",
                decision="allowed",
                resource_type="user",
                resource_id=user_id,
                reason=None,
                request_id=current.context.request_id,
                details={
                    "display_name_changed": before.display_name != after.display_name,
                    "status_before": before.status,
                    "status_after": after.status,
                    "revoked_count": revoked_count,
                },
                now=now,
            )
            return after

    def reset_password(
        self, actor: VerifiedIdentity, user_id: str, new_password: str
    ) -> None:
        password_hash = self._identity.hash_password(new_password)
        now = _now_text()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._require_actor(
                connection, actor, frozenset({IDENTITY_USER_MANAGE})
            )
            if self._identity_repository.get_user(connection, user_id) is None:
                raise ApiError(404, "user_not_found", "User not found")
            self._identity_repository.update_password(
                connection,
                user_id=user_id,
                password_hash=password_hash,
                now=now,
            )
            revoked_count = self._identity_repository.revoke_all_sessions(
                connection, user_id=user_id, now=now
            )
            self._audit.record(
                connection,
                actor_user_id=current.user_id,
                action="identity.user.password.reset",
                decision="allowed",
                resource_type="user",
                resource_id=user_id,
                reason=None,
                request_id=current.context.request_id,
                details={"revoked_count": revoked_count},
                now=now,
            )

    def list_capabilities(
        self, actor: VerifiedIdentity, *, limit: int, offset: int
    ) -> tuple[tuple[Capability, ...], int]:
        with connect(self._database_path) as connection:
            connection.execute("BEGIN")
            self._require_actor(connection, actor, frozenset({IDENTITY_ROLE_READ}))
            return (
                self._repository.list_capabilities(connection, limit=limit, offset=offset),
                self._repository.count_capabilities(connection),
            )

    def list_roles(
        self, actor: VerifiedIdentity, *, limit: int, offset: int
    ) -> tuple[tuple[Role, ...], int]:
        with connect(self._database_path) as connection:
            connection.execute("BEGIN")
            self._require_actor(connection, actor, frozenset({IDENTITY_ROLE_READ}))
            return (
                self._repository.list_roles(connection, limit=limit, offset=offset),
                self._repository.count_roles(connection),
            )

    def get_role(self, actor: VerifiedIdentity, role_id: str) -> Role:
        with connect(self._database_path) as connection:
            connection.execute("BEGIN")
            self._require_actor(connection, actor, frozenset({IDENTITY_ROLE_READ}))
            return self._require_role(connection, role_id)

    def create_role(
        self, actor: VerifiedIdentity, *, name: str, description: str
    ) -> Role:
        normalized_name = _validated_role_name(name)
        normalized_description = _validated_description(description)
        role_id = str(uuid.uuid4())
        now = _now_text()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._require_actor(
                connection, actor, frozenset({IDENTITY_ROLE_MANAGE})
            )
            try:
                self._repository.insert_role(
                    connection,
                    role_id=role_id,
                    name=normalized_name,
                    description=normalized_description,
                    now=now,
                )
            except sqlite3.IntegrityError as error:
                raise ApiError(409, "role_name_exists", "Role name already exists") from error
            self._audit.record(
                connection,
                actor_user_id=current.user_id,
                action="identity.role.create",
                decision="allowed",
                resource_type="role",
                resource_id=role_id,
                reason=None,
                request_id=current.context.request_id,
                details={"name": normalized_name},
                now=now,
            )
            return self._require_role(connection, role_id)

    def update_role(
        self,
        actor: VerifiedIdentity,
        role_id: str,
        *,
        name: str | None,
        description: str | None,
    ) -> Role:
        if name is None and description is None:
            raise ApiError(422, "invalid_request", "At least one role field is required")
        normalized_name = None if name is None else _validated_role_name(name)
        normalized_description = (
            None if description is None else _validated_description(description)
        )
        now = _now_text()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._require_actor(
                connection, actor, frozenset({IDENTITY_ROLE_MANAGE})
            )
            before = self._require_role(connection, role_id)
            self._reject_protected_role(before)
            try:
                self._repository.update_role(
                    connection,
                    role_id=role_id,
                    name=normalized_name or before.name,
                    description=(
                        before.description
                        if normalized_description is None
                        else normalized_description
                    ),
                    now=now,
                )
            except sqlite3.IntegrityError as error:
                raise ApiError(409, "role_name_exists", "Role name already exists") from error
            after = self._require_role(connection, role_id)
            self._audit.record(
                connection,
                actor_user_id=current.user_id,
                action="identity.role.update",
                decision="allowed",
                resource_type="role",
                resource_id=role_id,
                reason=None,
                request_id=current.context.request_id,
                details={"name_before": before.name, "name_after": after.name},
                now=now,
            )
            return after

    def delete_role(self, actor: VerifiedIdentity, role_id: str) -> None:
        now = _now_text()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._require_actor(
                connection, actor, frozenset({IDENTITY_ROLE_MANAGE})
            )
            role = self._require_role(connection, role_id)
            self._reject_protected_role(role)
            self._repository.delete_role(connection, role_id)
            self._audit.record(
                connection,
                actor_user_id=current.user_id,
                action="identity.role.delete",
                decision="allowed",
                resource_type="role",
                resource_id=role_id,
                reason=None,
                request_id=current.context.request_id,
                details={"name": role.name},
                now=now,
            )

    def replace_role_capabilities(
        self,
        actor: VerifiedIdentity,
        role_id: str,
        capability_codes: list[str],
    ) -> Role:
        requested = _unique_values(capability_codes, "capability_codes")
        now = _now_text()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._require_actor(
                connection, actor, frozenset({IDENTITY_ROLE_MANAGE})
            )
            role = self._require_role(connection, role_id)
            known = self._repository.all_capability_codes(connection)
            if not set(requested).issubset(known):
                raise ApiError(422, "unknown_capability", "Capability set contains an unknown code")
            if role.is_protected:
                if frozenset(requested) != ALL_CAPABILITIES:
                    raise ApiError(409, "protected_role", "Protected role cannot be changed")
            elif set(requested).intersection(ROOT_CAPABILITIES):
                raise ApiError(
                    403,
                    "root_capability_restricted",
                    "Security root capabilities are restricted to the protected system role",
                )
            before = role.capability_codes
            if not role.is_protected:
                self._repository.replace_role_capabilities(
                    connection,
                    role_id=role_id,
                    capability_codes=requested,
                    now=now,
                )
            self._audit.record(
                connection,
                actor_user_id=current.user_id,
                action="identity.role.capabilities.replace",
                decision="allowed",
                resource_type="role",
                resource_id=role_id,
                reason=None,
                request_id=current.context.request_id,
                details={"before": list(before), "after": list(requested)},
                now=now,
            )
            return self._require_role(connection, role_id)

    def replace_user_roles(
        self, actor: VerifiedIdentity, user_id: str, role_ids: list[str]
    ) -> tuple[str, ...]:
        requested = _unique_values(role_ids, "role_ids")
        now = _now_text()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._require_actor(
                connection, actor, frozenset({IDENTITY_ROLE_MANAGE})
            )
            if self._identity_repository.get_user(connection, user_id) is None:
                raise ApiError(404, "user_not_found", "User not found")
            if not self._repository.roles_exist(connection, requested):
                raise ApiError(422, "unknown_role", "Role set contains an unknown role")
            before = self._repository.user_role_ids(connection, user_id)
            self._repository.replace_user_roles(
                connection,
                user_id=user_id,
                role_ids=requested,
                assigned_by_user_id=current.user_id,
                now=now,
            )
            self._ensure_active_system_admin(connection)
            after = self._repository.user_role_ids(connection, user_id)
            self._audit.record(
                connection,
                actor_user_id=current.user_id,
                action="identity.user.roles.replace",
                decision="allowed",
                resource_type="user",
                resource_id=user_id,
                reason=None,
                request_id=current.context.request_id,
                details={"before": list(before), "after": list(after)},
                now=now,
            )
            return after

    def get_user_roles(
        self, actor: VerifiedIdentity, user_id: str
    ) -> tuple[str, ...]:
        with connect(self._database_path) as connection:
            connection.execute("BEGIN")
            self._require_actor(connection, actor, frozenset({IDENTITY_ROLE_READ}))
            if self._identity_repository.get_user(connection, user_id) is None:
                raise ApiError(404, "user_not_found", "User not found")
            return self._repository.user_role_ids(connection, user_id)

    def _require_actor(
        self,
        connection: sqlite3.Connection,
        actor: VerifiedIdentity,
        required: frozenset[str],
    ) -> VerifiedIdentity:
        current = self._identity.revalidate(actor.context, connection=connection)
        self._authorization.require_capabilities(
            current.user_id, required, connection=connection
        )
        return current

    def _insert_user(self, connection: sqlite3.Connection, prepared: PreparedUser) -> None:
        self._identity_repository.insert_user(
            connection,
            user_id=prepared.id,
            username=prepared.username,
            display_name=prepared.display_name,
            password_hash=prepared.password_hash,
            now=prepared.created_at,
        )

    def _require_role(self, connection: sqlite3.Connection, role_id: str) -> Role:
        role = self._repository.get_role(connection, role_id)
        if role is None:
            raise ApiError(404, "role_not_found", "Role not found")
        return role

    def _ensure_active_system_admin(self, connection: sqlite3.Connection) -> None:
        if self._repository.active_system_admin_count(connection) < 1:
            raise ApiError(
                409,
                "last_system_admin",
                "At least one active system administrator is required",
            )

    @staticmethod
    def _reject_protected_role(role: Role) -> None:
        if role.is_protected:
            raise ApiError(409, "protected_role", "Protected role cannot be changed")


def _unique_values(values: list[str], field: str) -> tuple[str, ...]:
    if len(values) > 100:
        raise ApiError(422, "invalid_request", f"{field} contains too many values")
    if len(values) != len(set(values)):
        raise ApiError(422, "duplicate_values", f"{field} contains duplicate values")
    return tuple(sorted(values))


def _validated_role_name(name: str) -> str:
    normalized = name.strip().lower()
    if not _ROLE_NAME.fullmatch(normalized):
        raise ApiError(422, "invalid_role_name", "Role name is invalid")
    return normalized


def _validated_description(description: str) -> str:
    normalized = description.strip()
    if len(normalized) > 500 or _has_control_character(normalized):
        raise ApiError(422, "invalid_description", "Role description is invalid")
    return normalized


def _has_control_character(value: str) -> bool:
    return any(unicodedata.category(character).startswith("C") for character in value)


def _now_text() -> str:
    return datetime.now(UTC).isoformat()
