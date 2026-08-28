from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from runfold_server.access_control.audit import AuditRepository
from runfold_server.access_control.authorization import AuthorizationService
from runfold_server.access_control.capabilities import (
    USAGE_ALL_READ,
    USAGE_LIMIT_MANAGE,
    USAGE_SELF_READ,
)
from runfold_server.errors import ApiError
from runfold_server.identity.models import VerifiedIdentity
from runfold_server.identity.repository import IdentityRepository
from runfold_server.identity.service import IdentityService
from runfold_server.storage.sqlite import connect
from runfold_server.usage.models import (
    EffectiveLimits,
    LimitOverrides,
    QuotaUsage,
    UsageSummary,
)
from runfold_server.usage.repository import UsageRepository


class UsageService:
    def __init__(
        self,
        *,
        database_path: Path,
        repository: UsageRepository,
        identity: IdentityService,
        identity_repository: IdentityRepository,
        authorization: AuthorizationService,
        audit: AuditRepository,
        default_max_documents: int,
        default_max_storage_bytes: int,
        default_monthly_embedding_tokens: int,
        default_monthly_agent_tokens: int,
    ) -> None:
        self._database_path = database_path
        self._repository = repository
        self._identity = identity
        self._identity_repository = identity_repository
        self._authorization = authorization
        self._audit = audit
        self._defaults = EffectiveLimits(
            max_documents=default_max_documents,
            max_storage_bytes=default_max_storage_bytes,
            monthly_embedding_tokens=default_monthly_embedding_tokens,
            monthly_agent_tokens=default_monthly_agent_tokens,
        )

    def limits(self, connection: sqlite3.Connection, user_id: str) -> EffectiveLimits:
        row = self._repository.overrides(connection, user_id)
        if row is None:
            return self._defaults
        return EffectiveLimits(
            max_documents=(
                self._defaults.max_documents
                if row["max_documents"] is None
                else int(row["max_documents"])
            ),
            max_storage_bytes=(
                self._defaults.max_storage_bytes
                if row["max_storage_bytes"] is None
                else int(row["max_storage_bytes"])
            ),
            monthly_embedding_tokens=(
                self._defaults.monthly_embedding_tokens
                if row["monthly_embedding_tokens"] is None
                else int(row["monthly_embedding_tokens"])
            ),
            monthly_agent_tokens=(
                self._defaults.monthly_agent_tokens
                if row["monthly_agent_tokens"] is None
                else int(row["monthly_agent_tokens"])
            ),
        )

    def require_create_capacity(
        self, connection: sqlite3.Connection, *, user_id: str, byte_size: int
    ) -> None:
        limits = self.limits(connection, user_id)
        documents, storage = self._repository.document_totals(connection, user_id)
        if documents >= limits.max_documents:
            raise _quota("documents")
        if storage + byte_size > limits.max_storage_bytes:
            raise _quota("storage_bytes")

    def require_creator_replace_capacity(
        self,
        connection: sqlite3.Connection,
        *,
        actor_user_id: str,
        creator_user_id: str,
        old_byte_size: int,
        new_byte_size: int,
    ) -> None:
        if actor_user_id != creator_user_id:
            return
        limits = self.limits(connection, creator_user_id)
        _, storage = self._repository.document_totals(connection, creator_user_id)
        if storage - old_byte_size + new_byte_size > limits.max_storage_bytes:
            raise _quota("storage_bytes")

    def require_embedding_capacity(
        self, connection: sqlite3.Connection, *, user_id: str
    ) -> None:
        limit = self.limits(connection, user_id).monthly_embedding_tokens
        current = self._repository.embedding_tokens(connection, user_id, _month_utc())
        if current >= limit:
            raise _quota("embedding_tokens")

    def record_embedding_tokens(self, user_id: str, tokens: int) -> None:
        now = datetime.now(UTC).isoformat()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._repository.add_embedding_tokens(
                connection,
                user_id=user_id,
                month_utc=_month_utc(),
                tokens=tokens,
                now=now,
            )

    def require_agent_capacity(
        self, connection: sqlite3.Connection, *, user_id: str
    ) -> None:
        limit = self.limits(connection, user_id).monthly_agent_tokens
        current = self._repository.agent_tokens(connection, user_id, _month_utc())
        if current >= limit:
            raise _quota("agent_tokens")

    def record_agent_tokens(self, user_id: str, tokens: int) -> None:
        if tokens <= 0:
            return
        now = datetime.now(UTC).isoformat()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._repository.add_agent_tokens(
                connection,
                user_id=user_id,
                month_utc=_month_utc(),
                tokens=tokens,
                now=now,
            )

    def record_upload(self, connection: sqlite3.Connection, user_id: str, now: str) -> None:
        self._repository.add_upload(
            connection, user_id=user_id, month_utc=_month_utc(), now=now
        )

    def record_search(self, connection: sqlite3.Connection, user_id: str, now: str) -> None:
        self._repository.add_search_request(
            connection, user_id=user_id, month_utc=_month_utc(), now=now
        )

    def self_summary(self, actor: VerifiedIdentity) -> UsageSummary:
        with connect(self._database_path) as connection:
            connection.execute("BEGIN")
            current = self._identity.revalidate(actor.context, connection=connection)
            self._authorization.require_capabilities(
                current.user_id,
                frozenset({USAGE_SELF_READ}),
                connection=connection,
            )
            return self._summary(connection, current.user_id)

    def user_summary(self, actor: VerifiedIdentity, user_id: str) -> UsageSummary:
        with connect(self._database_path) as connection:
            connection.execute("BEGIN")
            current = self._identity.revalidate(actor.context, connection=connection)
            self._authorization.require_capabilities(
                current.user_id,
                frozenset({USAGE_ALL_READ}),
                connection=connection,
            )
            self._require_user(connection, user_id)
            return self._summary(connection, user_id)

    def replace_limits(
        self,
        actor: VerifiedIdentity,
        user_id: str,
        overrides: LimitOverrides,
    ) -> UsageSummary:
        _validate_overrides(overrides)
        now = datetime.now(UTC).isoformat()
        with connect(self._database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._identity.revalidate(actor.context, connection=connection)
            self._authorization.require_capabilities(
                current.user_id,
                frozenset({USAGE_LIMIT_MANAGE}),
                connection=connection,
            )
            self._require_user(connection, user_id)
            before = self._raw_overrides(connection, user_id)
            self._repository.replace_overrides(
                connection,
                user_id=user_id,
                overrides=overrides,
                updated_by_user_id=current.user_id,
                now=now,
            )
            self._audit.record(
                connection,
                actor_user_id=current.user_id,
                action="usage.limits.replace",
                decision="allowed",
                resource_type="user",
                resource_id=user_id,
                reason=None,
                request_id=current.context.request_id,
                details={
                    "before": _safe_override_details(before),
                    "after": _safe_override_details(overrides),
                },
                now=now,
            )
            return self._summary(connection, user_id)

    def _summary(self, connection: sqlite3.Connection, user_id: str) -> UsageSummary:
        documents, storage = self._repository.document_totals(connection, user_id)
        month = _month_utc()
        embedding_tokens = self._repository.embedding_tokens(connection, user_id, month)
        agent_tokens = self._repository.agent_tokens(connection, user_id, month)
        limits = self.limits(connection, user_id)
        return UsageSummary(
            user_id=user_id,
            month_utc=month,
            documents=_quota_usage(documents, limits.max_documents),
            storage_bytes=_quota_usage(storage, limits.max_storage_bytes),
            embedding_tokens=_quota_usage(
                embedding_tokens, limits.monthly_embedding_tokens
            ),
            agent_tokens=_quota_usage(agent_tokens, limits.monthly_agent_tokens),
        )

    def _raw_overrides(
        self, connection: sqlite3.Connection, user_id: str
    ) -> LimitOverrides:
        row = self._repository.overrides(connection, user_id)
        if row is None:
            return LimitOverrides(None, None, None, None)
        return LimitOverrides(
            max_documents=(
                None if row["max_documents"] is None else int(row["max_documents"])
            ),
            max_storage_bytes=(
                None
                if row["max_storage_bytes"] is None
                else int(row["max_storage_bytes"])
            ),
            monthly_embedding_tokens=(
                None
                if row["monthly_embedding_tokens"] is None
                else int(row["monthly_embedding_tokens"])
            ),
            monthly_agent_tokens=(
                None
                if row["monthly_agent_tokens"] is None
                else int(row["monthly_agent_tokens"])
            ),
        )

    def _require_user(self, connection: sqlite3.Connection, user_id: str) -> None:
        if self._identity_repository.get_user(connection, user_id) is None:
            raise ApiError(404, "user_not_found", "User not found")


def _month_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


def _quota(name: str) -> ApiError:
    return ApiError(
        429,
        "quota_exceeded",
        "Usage quota exceeded",
        details={"quota": name},
    )


def _quota_usage(current: int, limit: int) -> QuotaUsage:
    return QuotaUsage(current=current, limit=limit, remaining=max(limit - current, 0))


def _validate_overrides(overrides: LimitOverrides) -> None:
    values = (
        overrides.max_documents,
        overrides.max_storage_bytes,
        overrides.monthly_embedding_tokens,
        overrides.monthly_agent_tokens,
    )
    if any(
        value is not None and (isinstance(value, bool) or value <= 0)
        for value in values
    ):
        raise ApiError(422, "invalid_limits", "Limits must be positive integers or null")


def _safe_override_details(overrides: LimitOverrides) -> dict[str, int | None]:
    return {
        "max_documents": overrides.max_documents,
        "max_storage_bytes": overrides.max_storage_bytes,
        "monthly_embedding_limit": overrides.monthly_embedding_tokens,
        "monthly_agent_limit": overrides.monthly_agent_tokens,
    }
