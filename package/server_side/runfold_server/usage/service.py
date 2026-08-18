from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from runfold_server.errors import ApiError
from runfold_server.storage.sqlite import connect
from runfold_server.usage.models import EffectiveLimits, UsageSummary
from runfold_server.usage.repository import UsageRepository


class UsageService:
    def __init__(
        self,
        *,
        database_path: Path,
        repository: UsageRepository,
        default_max_documents: int,
        default_max_storage_bytes: int,
        default_monthly_embedding_tokens: int,
    ) -> None:
        self._database_path = database_path
        self._repository = repository
        self._defaults = EffectiveLimits(
            max_documents=default_max_documents,
            max_storage_bytes=default_max_storage_bytes,
            monthly_embedding_tokens=default_monthly_embedding_tokens,
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

    def record_upload(self, connection: sqlite3.Connection, user_id: str, now: str) -> None:
        self._repository.add_upload(
            connection, user_id=user_id, month_utc=_month_utc(), now=now
        )

    def summary(self, user_id: str) -> UsageSummary:
        with connect(self._database_path) as connection:
            documents, storage = self._repository.document_totals(connection, user_id)
            return UsageSummary(
                document_count=documents,
                storage_bytes=storage,
                embedding_tokens=self._repository.embedding_tokens(
                    connection, user_id, _month_utc()
                ),
                limits=self.limits(connection, user_id),
            )


def _month_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


def _quota(name: str) -> ApiError:
    return ApiError(
        429,
        "quota_exceeded",
        "Usage quota exceeded",
        details={"quota": name},
    )
