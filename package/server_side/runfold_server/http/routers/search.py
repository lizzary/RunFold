from typing import Annotated

from fastapi import APIRouter, Depends

from runfold_server.http.auth_context import create_identity_dependency
from runfold_server.http.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
)
from runfold_server.identity.models import VerifiedIdentity
from runfold_server.identity.service import IdentityService
from runfold_server.knowledge.service import KnowledgeService


def create_search_router(
    identity_service: IdentityService, knowledge_service: KnowledgeService
) -> APIRouter:
    router = APIRouter(prefix="/api/rag", tags=["search"])
    current_identity = create_identity_dependency(identity_service)
    Actor = Annotated[VerifiedIdentity, Depends(current_identity)]

    @router.post("/search", response_model=SearchResponse)
    async def search(body: SearchRequest, actor: Actor) -> SearchResponse:
        document_ids = (
            None
            if "document_ids" not in body.model_fields_set
            else tuple(body.document_ids or ())
        )
        results = await knowledge_service.search(
            actor,
            query=body.query,
            top_k=body.top_k,
            document_ids=document_ids,
        )
        return SearchResponse(
            items=[SearchResultResponse.model_validate(result) for result in results]
        )

    return router
