from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from runfold_server.access_control.audit import AuditService
from runfold_server.http.auth_context import create_identity_dependency
from runfold_server.http.schemas.audit import AuditEventResponse, AuditEventsPage
from runfold_server.identity.models import VerifiedIdentity
from runfold_server.identity.service import IdentityService


def create_audit_router(
    identity_service: IdentityService, audit_service: AuditService
) -> APIRouter:
    router = APIRouter(prefix="/api/security", tags=["security"])
    current_identity = create_identity_dependency(identity_service)
    Actor = Annotated[VerifiedIdentity, Depends(current_identity)]
    Limit = Annotated[int, Query(ge=1, le=100)]
    Offset = Annotated[int, Query(ge=0)]
    Filter = Annotated[str | None, Query(min_length=1, max_length=128)]

    @router.get("/audit", response_model=AuditEventsPage)
    def list_audit_events(
        actor: Actor,
        limit: Limit = 50,
        offset: Offset = 0,
        actor_user_id: Filter = None,
        action: Filter = None,
        decision: Literal["allowed", "denied"] | None = None,
        resource_type: Filter = None,
        resource_id: Filter = None,
    ) -> AuditEventsPage:
        items, total = audit_service.list_events(
            actor,
            limit=limit,
            offset=offset,
            actor_user_id=actor_user_id,
            action=action,
            decision=decision,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        return AuditEventsPage(
            items=[AuditEventResponse.model_validate(item) for item in items],
            limit=limit,
            offset=offset,
            total=total,
        )

    return router
