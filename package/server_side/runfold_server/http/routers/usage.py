from typing import Annotated

from fastapi import APIRouter, Depends

from runfold_server.http.auth_context import create_identity_dependency
from runfold_server.http.schemas.usage import LimitsReplaceRequest, UsageResponse
from runfold_server.identity.models import VerifiedIdentity
from runfold_server.identity.service import IdentityService
from runfold_server.usage.models import LimitOverrides, UsageSummary
from runfold_server.usage.service import UsageService


def create_usage_router(
    identity_service: IdentityService, usage_service: UsageService
) -> APIRouter:
    router = APIRouter(prefix="/api/usage", tags=["usage"])
    current_identity = create_identity_dependency(identity_service)
    Actor = Annotated[VerifiedIdentity, Depends(current_identity)]

    @router.get("/me", response_model=UsageResponse)
    def own_usage(actor: Actor) -> UsageResponse:
        return _usage_response(usage_service.self_summary(actor))

    @router.get("/users/{user_id}", response_model=UsageResponse)
    def user_usage(user_id: str, actor: Actor) -> UsageResponse:
        return _usage_response(usage_service.user_summary(actor, user_id))

    @router.put("/users/{user_id}/limits", response_model=UsageResponse)
    def replace_limits(
        user_id: str, body: LimitsReplaceRequest, actor: Actor
    ) -> UsageResponse:
        summary = usage_service.replace_limits(
            actor,
            user_id,
            LimitOverrides(
                max_documents=body.max_documents,
                max_storage_bytes=body.max_storage_bytes,
                monthly_embedding_tokens=body.monthly_embedding_tokens,
            ),
        )
        return _usage_response(summary)

    return router


def _usage_response(summary: UsageSummary) -> UsageResponse:
    return UsageResponse.model_validate(summary)
