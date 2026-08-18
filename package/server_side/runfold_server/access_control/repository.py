from __future__ import annotations

import sqlite3

from runfold_server.access_control.models import Capability, CurrentAccess, Role


class AccessControlRepository:
    def list_capabilities(
        self, connection: sqlite3.Connection, *, limit: int, offset: int
    ) -> tuple[Capability, ...]:
        return tuple(
            Capability(code=str(row["code"]), description=str(row["description"]))
            for row in connection.execute(
                """
                SELECT code, description
                FROM capabilities
                ORDER BY code
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
        )

    def count_capabilities(self, connection: sqlite3.Connection) -> int:
        return int(connection.execute("SELECT COUNT(*) FROM capabilities").fetchone()[0])

    def list_roles(
        self, connection: sqlite3.Connection, *, limit: int, offset: int
    ) -> tuple[Role, ...]:
        rows = connection.execute(
            """
            SELECT id, name, description, is_protected, created_at, updated_at
            FROM roles
            ORDER BY name COLLATE NOCASE, id
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        return tuple(self._role(connection, row) for row in rows)

    def count_roles(self, connection: sqlite3.Connection) -> int:
        return int(connection.execute("SELECT COUNT(*) FROM roles").fetchone()[0])

    def get_role(self, connection: sqlite3.Connection, role_id: str) -> Role | None:
        row = connection.execute(
            """
            SELECT id, name, description, is_protected, created_at, updated_at
            FROM roles
            WHERE id = ?
            """,
            (role_id,),
        ).fetchone()
        return None if row is None else self._role(connection, row)

    def insert_role(
        self,
        connection: sqlite3.Connection,
        *,
        role_id: str,
        name: str,
        description: str,
        now: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO roles (id, name, description, is_protected, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?)
            """,
            (role_id, name, description, now, now),
        )

    def update_role(
        self,
        connection: sqlite3.Connection,
        *,
        role_id: str,
        name: str,
        description: str,
        now: str,
    ) -> None:
        connection.execute(
            "UPDATE roles SET name = ?, description = ?, updated_at = ? WHERE id = ?",
            (name, description, now, role_id),
        )

    def delete_role(self, connection: sqlite3.Connection, role_id: str) -> None:
        connection.execute("DELETE FROM roles WHERE id = ?", (role_id,))

    def replace_role_capabilities(
        self,
        connection: sqlite3.Connection,
        *,
        role_id: str,
        capability_codes: tuple[str, ...],
        now: str,
    ) -> None:
        connection.execute("DELETE FROM role_capabilities WHERE role_id = ?", (role_id,))
        connection.executemany(
            "INSERT INTO role_capabilities (role_id, capability_code) VALUES (?, ?)",
            ((role_id, code) for code in capability_codes),
        )
        connection.execute("UPDATE roles SET updated_at = ? WHERE id = ?", (now, role_id))

    def all_capability_codes(self, connection: sqlite3.Connection) -> frozenset[str]:
        return frozenset(
            str(row[0]) for row in connection.execute("SELECT code FROM capabilities")
        )

    def replace_user_roles(
        self,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        role_ids: tuple[str, ...],
        assigned_by_user_id: str | None,
        now: str,
    ) -> None:
        connection.execute("DELETE FROM user_roles WHERE user_id = ?", (user_id,))
        connection.executemany(
            """
            INSERT INTO user_roles (user_id, role_id, assigned_by_user_id, created_at)
            VALUES (?, ?, ?, ?)
            """,
            ((user_id, role_id, assigned_by_user_id, now) for role_id in role_ids),
        )

    def user_role_ids(self, connection: sqlite3.Connection, user_id: str) -> tuple[str, ...]:
        return tuple(
            str(row[0])
            for row in connection.execute(
                "SELECT role_id FROM user_roles WHERE user_id = ? ORDER BY role_id",
                (user_id,),
            )
        )

    def roles_exist(self, connection: sqlite3.Connection, role_ids: tuple[str, ...]) -> bool:
        if not role_ids:
            return True
        placeholders = ",".join("?" for _ in role_ids)
        count = connection.execute(
            f"SELECT COUNT(*) FROM roles WHERE id IN ({placeholders})", role_ids
        ).fetchone()[0]
        return int(count) == len(role_ids)

    def current_access(self, connection: sqlite3.Connection, user_id: str) -> CurrentAccess | None:
        user = connection.execute(
            "SELECT status FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if user is None or user["status"] != "active":
            return None
        role_ids = frozenset(
            str(row[0])
            for row in connection.execute(
                "SELECT role_id FROM user_roles WHERE user_id = ?", (user_id,)
            )
        )
        capabilities = frozenset(
            str(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT rc.capability_code
                FROM user_roles AS ur
                JOIN role_capabilities AS rc ON rc.role_id = ur.role_id
                WHERE ur.user_id = ?
                """,
                (user_id,),
            )
        )
        return CurrentAccess(user_id=user_id, role_ids=role_ids, capabilities=capabilities)

    def active_system_admin_count(self, connection: sqlite3.Connection) -> int:
        return int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM users AS u
                JOIN user_roles AS ur ON ur.user_id = u.id
                JOIN roles AS r ON r.id = ur.role_id
                WHERE u.status = 'active' AND r.is_protected = 1 AND r.name = 'system_admin'
                """
            ).fetchone()[0]
        )

    def _role(self, connection: sqlite3.Connection, row: sqlite3.Row) -> Role:
        capability_codes = tuple(
            str(item[0])
            for item in connection.execute(
                """
                SELECT capability_code
                FROM role_capabilities
                WHERE role_id = ?
                ORDER BY capability_code
                """,
                (row["id"],),
            )
        )
        return Role(
            id=str(row["id"]),
            name=str(row["name"]),
            description=str(row["description"]),
            is_protected=bool(row["is_protected"]),
            capability_codes=capability_codes,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
