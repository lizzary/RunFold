from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from runfold_server.access_control.authorization import AuthorizationService
from runfold_server.access_control.capabilities import SECURITY_AUDIT_READ
from runfold_server.access_control.models import AuditEvent
from runfold_server.storage.sqlite import connect

if TYPE_CHECKING:
    from runfold_server.identity.models import VerifiedIdentity
    from runfold_server.identity.service import IdentityService

_SENSITIVE_KEY = re.compile(
    r"(?i)(authorization|password|session|token|api.?key|content|chunk|query|prompt)"
)


class AuditRepository:
    def record(
        self,
        connection: sqlite3.Connection,
        *,
        actor_user_id: str | None,
        action: str,
        decision: str,
        resource_type: str,
        resource_id: str | None,
        reason: str | None,
        request_id: str,
        details: Mapping[str, object] | None,
        now: str,
    ) -> None:
        safe_details = dict(details or {})
        _assert_safe_details(safe_details)
        connection.execute(
            """
            INSERT INTO audit_events (
                actor_user_id, action, decision, resource_type, resource_id,
                reason, request_id, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                actor_user_id,
                action,
                decision,
                resource_type,
                resource_id,
                reason,
                request_id,
                json.dumps(safe_details, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                now,
            ),
        )

    def list_events(
        self,
        connection: sqlite3.Connection,
        *,
        limit: int,
        offset: int,
        actor_user_id: str | None,
        action: str | None,
        decision: str | None,
        resource_type: str | None,
        resource_id: str | None,
    ) -> tuple[AuditEvent, ...]:
        where, values = _event_filter(
            actor_user_id=actor_user_id,
            action=action,
            decision=decision,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        rows = connection.execute(
            f"""
            SELECT id, actor_user_id, action, decision, resource_type, resource_id,
                   reason, request_id, details_json, created_at
            FROM audit_events
            {where}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (*values, limit, offset),
        ).fetchall()
        return tuple(_event(row) for row in rows)

    def count_events(
        self,
        connection: sqlite3.Connection,
        *,
        actor_user_id: str | None,
        action: str | None,
        decision: str | None,
        resource_type: str | None,
        resource_id: str | None,
    ) -> int:
        where, values = _event_filter(
            actor_user_id=actor_user_id,
            action=action,
            decision=decision,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM audit_events {where}", values
            ).fetchone()[0]
        )


class AuditService:
    def __init__(
        self,
        *,
        database_path: Path,
        repository: AuditRepository,
        identity: IdentityService,
        authorization: AuthorizationService,
    ) -> None:
        self._database_path = database_path
        self._repository = repository
        self._identity = identity
        self._authorization = authorization

    def list_events(
        self,
        actor: VerifiedIdentity,
        *,
        limit: int,
        offset: int,
        actor_user_id: str | None,
        action: str | None,
        decision: str | None,
        resource_type: str | None,
        resource_id: str | None,
    ) -> tuple[tuple[AuditEvent, ...], int]:
        with connect(self._database_path) as connection:
            connection.execute("BEGIN")
            current = self._identity.revalidate(actor.context, connection=connection)
            self._authorization.require_capabilities(
                current.user_id,
                frozenset({SECURITY_AUDIT_READ}),
                connection=connection,
            )
            arguments = {
                "actor_user_id": actor_user_id,
                "action": action,
                "decision": decision,
                "resource_type": resource_type,
                "resource_id": resource_id,
            }
            return (
                self._repository.list_events(
                    connection, limit=limit, offset=offset, **arguments
                ),
                self._repository.count_events(connection, **arguments),
            )


def _assert_safe_details(value: object, key: str | None = None) -> None:
    if key is not None and _SENSITIVE_KEY.search(key):
        raise ValueError("Sensitive values are not permitted in audit details")
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            if not isinstance(child_key, str):
                raise ValueError("Audit detail keys must be strings")
            _assert_safe_details(child_value, child_key)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_safe_details(child)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError("Unsupported audit detail value")


def _event_filter(
    *,
    actor_user_id: str | None,
    action: str | None,
    decision: str | None,
    resource_type: str | None,
    resource_id: str | None,
) -> tuple[str, tuple[object, ...]]:
    clauses: list[str] = []
    values: list[object] = []
    for column, value in (
        ("actor_user_id", actor_user_id),
        ("action", action),
        ("decision", decision),
        ("resource_type", resource_type),
        ("resource_id", resource_id),
    ):
        if value is not None:
            clauses.append(f"{column} = ?")
            values.append(value)
    return ("WHERE " + " AND ".join(clauses) if clauses else "", tuple(values))


def _event(row: sqlite3.Row) -> AuditEvent:
    details = json.loads(str(row["details_json"]))
    if not isinstance(details, dict):
        raise ValueError("Audit details must be an object")
    return AuditEvent(
        id=int(row["id"]),
        actor_user_id=(
            None if row["actor_user_id"] is None else str(row["actor_user_id"])
        ),
        action=str(row["action"]),
        decision=str(row["decision"]),
        resource_type=str(row["resource_type"]),
        resource_id=None if row["resource_id"] is None else str(row["resource_id"]),
        reason=None if row["reason"] is None else str(row["reason"]),
        request_id=str(row["request_id"]),
        details=details,
        created_at=str(row["created_at"]),
    )
