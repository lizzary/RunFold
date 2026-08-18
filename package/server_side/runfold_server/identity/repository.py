from __future__ import annotations

import sqlite3

from runfold_server.identity.models import User


class IdentityRepository:
    def count_users(self, connection: sqlite3.Connection) -> int:
        return int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def insert_user(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        username: str,
        display_name: str,
        password_hash: str,
        now: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO users (
                id, username, display_name, password_hash, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'active', ?, ?)
            """,
            (user_id, username, display_name, password_hash, now, now),
        )

    def get_user(self, connection: sqlite3.Connection, user_id: str) -> User | None:
        row = connection.execute(
            """
            SELECT id, username, display_name, status, created_at, updated_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
        return _user(row)

    def get_user_with_password_by_username(
        self, connection: sqlite3.Connection, username: str
    ) -> tuple[User, str] | None:
        row = connection.execute(
            """
            SELECT id, username, display_name, password_hash, status, created_at, updated_at
            FROM users
            WHERE username = ? COLLATE NOCASE
            """,
            (username,),
        ).fetchone()
        if row is None:
            return None
        return _user(row), str(row["password_hash"])

    def get_user_with_password(
        self, connection: sqlite3.Connection, user_id: str
    ) -> tuple[User, str] | None:
        row = connection.execute(
            """
            SELECT id, username, display_name, password_hash, status, created_at, updated_at
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return _user(row), str(row["password_hash"])

    def list_users(
        self, connection: sqlite3.Connection, *, limit: int, offset: int
    ) -> tuple[User, ...]:
        return tuple(
            _user(row)
            for row in connection.execute(
                """
                SELECT id, username, display_name, status, created_at, updated_at
                FROM users
                ORDER BY username COLLATE NOCASE, id
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
        )

    def update_user(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        display_name: str,
        status: str,
        now: str,
    ) -> None:
        connection.execute(
            """
            UPDATE users
            SET display_name = ?, status = ?, updated_at = ?
            WHERE id = ?
            """,
            (display_name, status, now, user_id),
        )

    def update_password(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        password_hash: str,
        now: str,
    ) -> None:
        connection.execute(
            "UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?",
            (password_hash, now, user_id),
        )

    def insert_session(
        self,
        connection: sqlite3.Connection,
        *,
        session_id: str,
        user_id: str,
        token_hash: str,
        expires_at: str,
        now: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO auth_sessions (
                id, user_id, token_hash, expires_at, revoked_at, created_at
            ) VALUES (?, ?, ?, ?, NULL, ?)
            """,
            (session_id, user_id, token_hash, expires_at, now),
        )

    def get_session_identity_by_hash(
        self, connection: sqlite3.Connection, token_hash: str
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT
                s.id AS session_id,
                s.user_id,
                s.expires_at,
                s.revoked_at,
                u.username,
                u.display_name,
                u.status,
                u.created_at,
                u.updated_at
            FROM auth_sessions AS s
            JOIN users AS u ON u.id = s.user_id
            WHERE s.token_hash = ?
            """,
            (token_hash,),
        ).fetchone()

    def get_session_identity(
        self, connection: sqlite3.Connection, *, session_id: str, user_id: str
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT
                s.id AS session_id,
                s.user_id,
                s.expires_at,
                s.revoked_at,
                u.username,
                u.display_name,
                u.status,
                u.created_at,
                u.updated_at
            FROM auth_sessions AS s
            JOIN users AS u ON u.id = s.user_id
            WHERE s.id = ? AND s.user_id = ?
            """,
            (session_id, user_id),
        ).fetchone()

    def revoke_session(
        self, connection: sqlite3.Connection, *, session_id: str, now: str
    ) -> None:
        connection.execute(
            "UPDATE auth_sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
            (now, session_id),
        )

    def revoke_all_sessions(
        self, connection: sqlite3.Connection, *, user_id: str, now: str
    ) -> int:
        cursor = connection.execute(
            "UPDATE auth_sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
            (now, user_id),
        )
        return cursor.rowcount


def _user(row: sqlite3.Row | None) -> User | None:
    if row is None:
        return None
    return User(
        id=str(row["id"]),
        username=str(row["username"]),
        display_name=str(row["display_name"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
