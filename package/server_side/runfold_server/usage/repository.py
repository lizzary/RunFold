from __future__ import annotations

import sqlite3


class UsageRepository:
    def overrides(self, connection: sqlite3.Connection, user_id: str) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT max_documents, max_storage_bytes, monthly_embedding_tokens
            FROM user_limits
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

    def document_totals(self, connection: sqlite3.Connection, user_id: str) -> tuple[int, int]:
        row = connection.execute(
            """
            SELECT COUNT(*) AS document_count, COALESCE(SUM(byte_size), 0) AS storage_bytes
            FROM documents
            WHERE created_by_user_id = ?
            """,
            (user_id,),
        ).fetchone()
        return int(row["document_count"]), int(row["storage_bytes"])

    def embedding_tokens(
        self, connection: sqlite3.Connection, user_id: str, month_utc: str
    ) -> int:
        row = connection.execute(
            """
            SELECT embedding_tokens
            FROM usage_monthly
            WHERE user_id = ? AND month_utc = ?
            """,
            (user_id, month_utc),
        ).fetchone()
        return 0 if row is None else int(row[0])

    def add_embedding_tokens(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        month_utc: str,
        tokens: int,
        now: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO usage_monthly (
                user_id, month_utc, embedding_tokens, uploads, updated_at
            ) VALUES (?, ?, ?, 0, ?)
            ON CONFLICT(user_id, month_utc) DO UPDATE SET
                embedding_tokens = embedding_tokens + excluded.embedding_tokens,
                updated_at = excluded.updated_at
            """,
            (user_id, month_utc, tokens, now),
        )

    def add_upload(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        month_utc: str,
        now: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO usage_monthly (
                user_id, month_utc, embedding_tokens, uploads, updated_at
            ) VALUES (?, ?, 0, 1, ?)
            ON CONFLICT(user_id, month_utc) DO UPDATE SET
                uploads = uploads + 1,
                updated_at = excluded.updated_at
            """,
            (user_id, month_utc, now),
        )
