from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping

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
