from __future__ import annotations

import sqlite3

from runfold_server.access_control.models import CurrentAccess
from runfold_server.knowledge.models import AclGrant, Document


class KnowledgeRepository:
    def insert_document(
        self,
        connection: sqlite3.Connection,
        *,
        document_id: str,
        title: str,
        creator_user_id: str,
        original_filename: str,
        media_type: str,
        byte_size: int,
        content_hash: str,
        extracted_characters: int,
        chunk_count: int,
        now: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO documents (
                id, title, created_by_user_id, original_filename, media_type,
                storage_key, byte_size, content_hash, extracted_characters,
                chunk_count, index_state, index_error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'indexing', NULL, ?, ?)
            """,
            (
                document_id,
                title,
                creator_user_id,
                original_filename,
                media_type,
                f"{document_id}/source",
                byte_size,
                content_hash,
                extracted_characters,
                chunk_count,
                now,
                now,
            ),
        )

    def insert_creator_acl(
        self,
        connection: sqlite3.Connection,
        *,
        document_id: str,
        user_id: str,
        access_level: int,
        now: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO document_acl (
                document_id, user_id, role_id, access_level, granted_by_user_id, created_at
            ) VALUES (?, ?, NULL, ?, ?, ?)
            """,
            (document_id, user_id, access_level, user_id, now),
        )

    def get(self, connection: sqlite3.Connection, document_id: str) -> Document | None:
        row = connection.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        return _document(row)

    def get_authorized(
        self,
        connection: sqlite3.Connection,
        *,
        access: CurrentAccess,
        document_id: str,
        minimum_level: int,
        states: tuple[str, ...],
    ) -> Document | None:
        state_placeholders = ",".join("?" for _ in states)
        acl_sql, acl_values = _acl_exists(access, minimum_level)
        row = connection.execute(
            f"""
            SELECT d.*
            FROM documents AS d
            WHERE d.id = ?
              AND d.index_state IN ({state_placeholders})
              AND {acl_sql}
            """,
            (document_id, *states, *acl_values),
        ).fetchone()
        return _document(row)

    def list_authorized(
        self,
        connection: sqlite3.Connection,
        *,
        access: CurrentAccess,
        minimum_level: int,
        states: tuple[str, ...],
        limit: int,
        offset: int,
    ) -> tuple[Document, ...]:
        state_placeholders = ",".join("?" for _ in states)
        if access.bypass:
            sql = f"""
                SELECT d.* FROM documents AS d
                WHERE d.index_state IN ({state_placeholders})
                ORDER BY d.created_at DESC, d.id
                LIMIT ? OFFSET ?
            """
            values: tuple[object, ...] = (*states, limit, offset)
        else:
            acl_sql, acl_values = _acl_exists(access, minimum_level)
            sql = f"""
                SELECT d.* FROM documents AS d
                WHERE d.index_state IN ({state_placeholders}) AND {acl_sql}
                ORDER BY d.created_at DESC, d.id
                LIMIT ? OFFSET ?
            """
            values = (*states, *acl_values, limit, offset)
        return tuple(_document(row) for row in connection.execute(sql, values))

    def count_authorized(
        self,
        connection: sqlite3.Connection,
        *,
        access: CurrentAccess,
        minimum_level: int,
        states: tuple[str, ...],
    ) -> int:
        state_placeholders = ",".join("?" for _ in states)
        if access.bypass:
            row = connection.execute(
                f"SELECT COUNT(*) FROM documents d WHERE d.index_state IN ({state_placeholders})",
                states,
            ).fetchone()
        else:
            acl_sql, acl_values = _acl_exists(access, minimum_level)
            row = connection.execute(
                f"""
                SELECT COUNT(*) FROM documents d
                WHERE d.index_state IN ({state_placeholders}) AND {acl_sql}
                """,
                (*states, *acl_values),
            ).fetchone()
        return int(row[0])

    def update_title(
        self, connection: sqlite3.Connection, document_id: str, title: str, now: str
    ) -> None:
        connection.execute(
            "UPDATE documents SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, document_id),
        )

    def transition_to_indexing(
        self,
        connection: sqlite3.Connection,
        *,
        document_id: str,
        observed_hash: str,
        now: str,
    ) -> bool:
        cursor = connection.execute(
            """
            UPDATE documents
            SET index_state = 'indexing', index_error = NULL, updated_at = ?
            WHERE id = ? AND content_hash = ? AND index_state IN ('ready', 'failed')
            """,
            (now, document_id, observed_hash),
        )
        return cursor.rowcount == 1

    def update_indexing_content(
        self,
        connection: sqlite3.Connection,
        *,
        document_id: str,
        original_filename: str,
        media_type: str,
        byte_size: int,
        content_hash: str,
        extracted_characters: int,
        chunk_count: int,
        now: str,
    ) -> bool:
        cursor = connection.execute(
            """
            UPDATE documents
            SET original_filename = ?, media_type = ?, byte_size = ?, content_hash = ?,
                extracted_characters = ?, chunk_count = ?, index_error = NULL, updated_at = ?
            WHERE id = ? AND index_state = 'indexing'
            """,
            (
                original_filename,
                media_type,
                byte_size,
                content_hash,
                extracted_characters,
                chunk_count,
                now,
                document_id,
            ),
        )
        return cursor.rowcount == 1

    def mark_ready(
        self,
        connection: sqlite3.Connection,
        *,
        document_id: str,
        content_hash: str,
        now: str,
    ) -> bool:
        cursor = connection.execute(
            """
            UPDATE documents
            SET index_state = 'ready', index_error = NULL, updated_at = ?
            WHERE id = ? AND index_state = 'indexing' AND content_hash = ?
            """,
            (now, document_id, content_hash),
        )
        return cursor.rowcount == 1

    def mark_failed(
        self,
        connection: sqlite3.Connection,
        *,
        document_id: str,
        original_filename: str,
        media_type: str,
        byte_size: int,
        content_hash: str,
        error_code: str,
        now: str,
    ) -> None:
        connection.execute(
            """
            UPDATE documents
            SET original_filename = ?, media_type = ?, byte_size = ?, content_hash = ?,
                extracted_characters = 0, chunk_count = 0, index_state = 'failed',
                index_error = ?, updated_at = ?
            WHERE id = ? AND index_state = 'indexing'
            """,
            (
                original_filename,
                media_type,
                byte_size,
                content_hash,
                error_code,
                now,
                document_id,
            ),
        )

    def transition_to_deleting(
        self,
        connection: sqlite3.Connection,
        *,
        document_id: str,
        observed_hash: str,
        now: str,
    ) -> bool:
        cursor = connection.execute(
            """
            UPDATE documents SET index_state = 'deleting', updated_at = ?
            WHERE id = ? AND content_hash = ? AND index_state IN ('ready', 'failed')
            """,
            (now, document_id, observed_hash),
        )
        return cursor.rowcount == 1

    def delete_record(self, connection: sqlite3.Connection, document_id: str) -> None:
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def documents_in_states(
        self, connection: sqlite3.Connection, states: tuple[str, ...]
    ) -> tuple[Document, ...]:
        placeholders = ",".join("?" for _ in states)
        return tuple(
            _document(row)
            for row in connection.execute(
                f"SELECT * FROM documents WHERE index_state IN ({placeholders}) ORDER BY id",
                states,
            )
        )

    def list_acl(
        self, connection: sqlite3.Connection, document_id: str
    ) -> tuple[AclGrant, ...]:
        return tuple(
            AclGrant(
                user_id=None if row["user_id"] is None else str(row["user_id"]),
                role_id=None if row["role_id"] is None else str(row["role_id"]),
                access_level=int(row["access_level"]),
            )
            for row in connection.execute(
                """
                SELECT user_id, role_id, access_level
                FROM document_acl
                WHERE document_id = ?
                ORDER BY user_id, role_id
                """,
                (document_id,),
            )
        )

    def replace_acl(
        self,
        connection: sqlite3.Connection,
        *,
        document_id: str,
        grants: tuple[AclGrant, ...],
        granted_by_user_id: str,
        now: str,
    ) -> None:
        connection.execute("DELETE FROM document_acl WHERE document_id = ?", (document_id,))
        connection.executemany(
            """
            INSERT INTO document_acl (
                document_id, user_id, role_id, access_level,
                granted_by_user_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    document_id,
                    grant.user_id,
                    grant.role_id,
                    grant.access_level,
                    granted_by_user_id,
                    now,
                )
                for grant in grants
            ),
        )

    def acl_subjects_exist(
        self, connection: sqlite3.Connection, grants: tuple[AclGrant, ...]
    ) -> bool:
        for grant in grants:
            if grant.user_id is not None:
                row = connection.execute(
                    "SELECT 1 FROM users WHERE id = ?", (grant.user_id,)
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT 1 FROM roles WHERE id = ?", (grant.role_id,)
                ).fetchone()
            if row is None:
                return False
        return True


def _acl_exists(access: CurrentAccess, minimum_level: int) -> tuple[str, tuple[object, ...]]:
    role_ids = tuple(sorted(access.role_ids))
    if role_ids:
        placeholders = ",".join("?" for _ in role_ids)
        subject = f"(a.user_id = ? OR a.role_id IN ({placeholders}))"
        values: tuple[object, ...] = (access.user_id, *role_ids, minimum_level)
    else:
        subject = "a.user_id = ?"
        values = (access.user_id, minimum_level)
    return (
        f"""EXISTS (
            SELECT 1 FROM document_acl AS a
            WHERE a.document_id = d.id AND {subject} AND a.access_level >= ?
        )""",
        values,
    )


def _document(row: sqlite3.Row | None) -> Document | None:
    if row is None:
        return None
    return Document(
        id=str(row["id"]),
        title=str(row["title"]),
        created_by_user_id=str(row["created_by_user_id"]),
        original_filename=str(row["original_filename"]),
        media_type=str(row["media_type"]),
        storage_key=str(row["storage_key"]),
        byte_size=int(row["byte_size"]),
        content_hash=str(row["content_hash"]),
        extracted_characters=int(row["extracted_characters"]),
        chunk_count=int(row["chunk_count"]),
        index_state=str(row["index_state"]),
        index_error=None if row["index_error"] is None else str(row["index_error"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
