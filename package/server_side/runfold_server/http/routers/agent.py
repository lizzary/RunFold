from typing import Annotated

from fastapi import APIRouter, Depends

from runfold_server.http.auth_context import create_identity_dependency
from runfold_server.http.schemas.agent import AgentRunRequest, AgentRunResponse
from runfold_server.identity.models import VerifiedIdentity
from runfold_server.identity.service import IdentityService
from runfold_server.runtime.service import AgentRuntimeService


def create_agent_router(
    identity_service: IdentityService,
    runtime_service: AgentRuntimeService,
) -> APIRouter:
    router = APIRouter(prefix="/api/agent", tags=["agent"])
    current_identity = create_identity_dependency(identity_service)
    Actor = Annotated[VerifiedIdentity, Depends(current_identity)]

    @router.post("/runs", response_model=AgentRunResponse)
    async def run_agent(body: AgentRunRequest, actor: Actor) -> AgentRunResponse:
        result = await runtime_service.run(actor, body.input)
        return AgentRunResponse.model_validate(result)

    return router
