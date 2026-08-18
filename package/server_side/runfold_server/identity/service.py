from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import unicodedata
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from runfold_server.access_control.audit import AuditRepository
from runfold_server.errors import ApiError
from runfold_server.identity.models import (
    AuthContext,
    LoginResult,
    PreparedUser,
    User,
    VerifiedIdentity,
)
from runfold_server.identity.passwords import Argon2PasswordHasher
from runfold_server.identity.repository import IdentityRepository
from runfold_server.storage.sqlite import connect

_USERNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,63}$")


class IdentityService:
    def __init__(
        self,
        *,
        database_path: Path,
        repository: IdentityRepository,
        password_hasher: Argon2PasswordHasher,
        audit: AuditRepository,
        session_ttl_seconds: int,
    ) -> None:
        self._database_path = database_path
        self._repository = repository
        self._password_hasher = password_hasher
        self._audit = audit
        self._session_ttl = timedelta(seconds=session_ttl_seconds)

    def prepare_user(self, username: str, display_name: str, password: str) -> PreparedUser:
        normalized_username = _validated_username(username)
        normalized_display_name = self.normalize_display_name(display_name)
        return PreparedUser(
            id=str(uuid.uuid4()),
            username=normalized_username,
            display_name=normalized_display_name,
            password_hash=self.hash_password(password),
            created_at=_now_text(),
        )

    def normalize_display_name(self, display_name: str) -> str:
        return _validated_display_name(display_name)

    def hash_password(self, password: str) -> str:
        _validate_password(password)
        return self._password_hasher.hash(password)

    def login(self, username: str, password: str, request_id: str) -> LoginResult:
        normalized_username = username.strip().lower()
        with connect(self._database_path) as connection:
            record = self._repository.get_user_with_password_by_username(
                connection, normalized_username
            )

        if record is None:
            self._password_hasher.verify_dummy(password)
            self._record_login_denied(normalized_username, request_id)
            raise ApiError(401, "invalid_credentials", "Invalid username or password")

        user, observed_hash = record
        password_matches = self._password_hasher.verify(observed_hash, password)
        if not password_matches or user.status != "active":
            self._record_login_denied(normalized_username, request_id)
            raise ApiError(401, "invalid_credentials", "Invalid username or password")

        token = secrets.token_urlsafe(32)
        token_hash = _session_hash(token)
        session_id = str(uuid.uuid4())
        now = _now()
        now_text = now.isoformat()
        expires_at = (now + self._session_ttl).isoformat()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._repository.get_user_with_password(connection, user.id)
            if current is None or current[0].status != "active" or current[1] != observed_hash:
                self._audit.record(
                    connection,
                    actor_user_id=None,
                    action="auth.login",
                    decision="denied",
                    resource_type="identity",
                    resource_id=None,
                    reason="invalid_credentials",
                    request_id=request_id,
                    details={"username": normalized_username},
                    now=now_text,
                )
                connection.commit()
                raise ApiError(401, "invalid_credentials", "Invalid username or password")
            current_user = current[0]
            self._repository.insert_session(
                connection,
                session_id=session_id,
                user_id=current_user.id,
                token_hash=token_hash,
                expires_at=expires_at,
                now=now_text,
            )
            self._audit.record(
                connection,
                actor_user_id=current_user.id,
                action="auth.login",
                decision="allowed",
                resource_type="user",
                resource_id=current_user.id,
                reason=None,
                request_id=request_id,
                details={},
                now=now_text,
            )
        context = AuthContext(
            user_id=current_user.id,
            session_id=session_id,
            request_id=request_id,
        )
        return LoginResult(
            token=token,
            expires_at=expires_at,
            identity=VerifiedIdentity(context=context, user=current_user),
        )

    def authenticate(self, token: str, request_id: str) -> VerifiedIdentity:
        if not token or len(token) > 512:
            raise _invalid_session()
        with connect(self._database_path) as connection:
            row = self._repository.get_session_identity_by_hash(
                connection, _session_hash(token)
            )
            return self._verified_from_session_row(row, request_id)

    def revalidate(
        self,
        context: AuthContext,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> VerifiedIdentity:
        if connection is None:
            with connect(self._database_path) as current_connection:
                return self.revalidate(context, connection=current_connection)
        row = self._repository.get_session_identity(
            connection,
            session_id=context.session_id,
            user_id=context.user_id,
        )
        return self._verified_from_session_row(row, context.request_id)

    def logout(self, identity: VerifiedIdentity) -> None:
        now = _now_text()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self.revalidate(identity.context, connection=connection)
            self._repository.revoke_session(
                connection, session_id=current.context.session_id, now=now
            )
            self._audit.record(
                connection,
                actor_user_id=current.user_id,
                action="auth.logout",
                decision="allowed",
                resource_type="user",
                resource_id=current.user_id,
                reason=None,
                request_id=current.context.request_id,
                details={},
                now=now,
            )

    def change_password(
        self, identity: VerifiedIdentity, current_password: str, new_password: str
    ) -> None:
        with connect(self._database_path) as connection:
            observed = self._repository.get_user_with_password(connection, identity.user_id)
        if observed is None or not self._password_hasher.verify(observed[1], current_password):
            raise ApiError(400, "invalid_current_password", "Current password is invalid")
        new_hash = self.hash_password(new_password)
        now = _now_text()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self.revalidate(identity.context, connection=connection)
            current_record = self._repository.get_user_with_password(
                connection, current.user_id
            )
            if current_record is None or current_record[1] != observed[1]:
                raise ApiError(409, "identity_changed", "Identity changed; retry the request")
            self._repository.update_password(
                connection,
                user_id=current.user_id,
                password_hash=new_hash,
                now=now,
            )
            revoked_count = self._repository.revoke_all_sessions(
                connection, user_id=current.user_id, now=now
            )
            self._audit.record(
                connection,
                actor_user_id=current.user_id,
                action="auth.password.change",
                decision="allowed",
                resource_type="user",
                resource_id=current.user_id,
                reason=None,
                request_id=current.context.request_id,
                details={"revoked_count": revoked_count},
                now=now,
            )

    def _verified_from_session_row(
        self, row: sqlite3.Row | None, request_id: str
    ) -> VerifiedIdentity:
        if (
            row is None
            or row["revoked_at"] is not None
            or row["status"] != "active"
            or _parse_time(str(row["expires_at"])) <= _now()
        ):
            raise _invalid_session()
        user = User(
            id=str(row["user_id"]),
            username=str(row["username"]),
            display_name=str(row["display_name"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
        return VerifiedIdentity(
            context=AuthContext(
                user_id=user.id,
                session_id=str(row["session_id"]),
                request_id=request_id,
            ),
            user=user,
        )

    def _record_login_denied(self, username: str, request_id: str) -> None:
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._audit.record(
                connection,
                actor_user_id=None,
                action="auth.login",
                decision="denied",
                resource_type="identity",
                resource_id=None,
                reason="invalid_credentials",
                request_id=request_id,
                details={"username": username[:64]},
                now=_now_text(),
            )


def _validated_username(username: str) -> str:
    normalized = username.strip().lower()
    if not _USERNAME.fullmatch(normalized):
        raise ApiError(
            422,
            "invalid_username",
            "Username must be 3-64 ASCII letters, digits, dots, underscores, or hyphens",
        )
    return normalized


def _validated_display_name(display_name: str) -> str:
    normalized = display_name.strip()
    if not normalized or len(normalized) > 100 or _has_control_character(normalized):
        raise ApiError(422, "invalid_display_name", "Display name is invalid")
    return normalized


def _validate_password(password: str) -> None:
    if len(password) < 12 or len(password) > 256 or _has_control_character(password):
        raise ApiError(
            422,
            "invalid_password",
            "Password must contain 12-256 characters and no control characters",
        )


def _has_control_character(value: str) -> bool:
    return any(unicodedata.category(character).startswith("C") for character in value)


def _session_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(UTC)


def _now_text() -> str:
    return _now().isoformat()


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise _invalid_session() from None
    if parsed.tzinfo is None:
        raise _invalid_session()
    return parsed.astimezone(UTC)


def _invalid_session() -> ApiError:
    return ApiError(401, "invalid_session", "Authentication is required")
