from __future__ import annotations

from typing import Any, Literal

from runfold_server.http.schemas.auth import StrictModel


class AuditEventResponse(StrictModel):
    id: int
    actor_user_id: str | None
    action: str
    decision: Literal["allowed", "denied"]
    resource_type: str
    resource_id: str | None
    reason: str | None
    request_id: str
    details: dict[str, Any]
    created_at: str


class AuditEventsPage(StrictModel):
    items: list[AuditEventResponse]
    limit: int
    offset: int
    total: int
