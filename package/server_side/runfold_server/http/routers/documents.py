from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile, status

from runfold_server.errors import ApiError
from runfold_server.http.auth_context import create_identity_dependency
from runfold_server.http.schemas.documents import (
    AclGrantResponse,
    DocumentAclReplaceRequest,
    DocumentAclResponse,
    DocumentMetadataUpdateRequest,
    DocumentResponse,
    DocumentsPage,
    DocumentTextReplaceRequest,
    DocumentTextResponse,
)
from runfold_server.identity.models import VerifiedIdentity
from runfold_server.identity.service import IdentityService
from runfold_server.knowledge.models import AclGrant, Document
from runfold_server.knowledge.service import KnowledgeService


def create_documents_router(
    identity_service: IdentityService, knowledge_service: KnowledgeService
) -> APIRouter:
    router = APIRouter(prefix="/api/rag/documents", tags=["documents"])
    current_identity = create_identity_dependency(identity_service)
    Actor = Annotated[VerifiedIdentity, Depends(current_identity)]
    Limit = Annotated[int, Query(ge=1, le=100)]
    Offset = Annotated[int, Query(ge=0)]

    @router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
    async def upload_document(
        request: Request,
        actor: Actor,
        file: Annotated[UploadFile, File()],
        title: Annotated[str, Form(min_length=1, max_length=200)],
    ) -> DocumentResponse:
        await _require_multipart_fields(request, {"file": 1, "title": 1})
        document = await knowledge_service.upload(
            actor,
            title=title,
            original_filename=file.filename or "",
            stream=file,
        )
        return _document_response(document)

    @router.get("", response_model=DocumentsPage)
    def list_documents(actor: Actor, limit: Limit = 50, offset: Offset = 0) -> DocumentsPage:
        items, total = knowledge_service.list_documents(actor, limit=limit, offset=offset)
        return DocumentsPage(
            items=[_document_response(item) for item in items],
            limit=limit,
            offset=offset,
            total=total,
        )

    @router.get("/{document_id}", response_model=DocumentResponse)
    def get_document(document_id: str, actor: Actor) -> DocumentResponse:
        return _document_response(knowledge_service.get_document(actor, document_id))

    @router.patch("/{document_id}", response_model=DocumentResponse)
    def update_document(
        document_id: str, body: DocumentMetadataUpdateRequest, actor: Actor
    ) -> DocumentResponse:
        return _document_response(
            knowledge_service.update_metadata(actor, document_id, title=body.title)
        )

    @router.get("/{document_id}/content")
    def download_document(document_id: str, actor: Actor) -> Response:
        value = knowledge_service.download(actor, document_id)
        encoded = quote(value.original_filename, safe="")
        return Response(
            content=value.data,
            media_type=value.media_type,
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}"},
        )

    @router.get("/{document_id}/text", response_model=DocumentTextResponse)
    def get_document_text(document_id: str, actor: Actor) -> DocumentTextResponse:
        value = knowledge_service.extracted_text(actor, document_id)
        return DocumentTextResponse.model_validate(value)

    @router.put("/{document_id}/content", response_model=DocumentResponse)
    async def replace_document_content(
        document_id: str,
        request: Request,
        actor: Actor,
        file: Annotated[UploadFile, File()],
    ) -> DocumentResponse:
        await _require_multipart_fields(request, {"file": 1})
        document = await knowledge_service.replace_upload(
            actor,
            document_id,
            original_filename=file.filename or "",
            stream=file,
        )
        return _document_response(document)

    @router.put("/{document_id}/text", response_model=DocumentResponse)
    async def replace_document_text(
        document_id: str, body: DocumentTextReplaceRequest, actor: Actor
    ) -> DocumentResponse:
        return _document_response(
            await knowledge_service.replace_text(
                actor, document_id, text=body.text
            )
        )

    @router.post("/{document_id}/reindex", response_model=DocumentResponse)
    async def reindex_document(document_id: str, actor: Actor) -> DocumentResponse:
        return _document_response(await knowledge_service.reindex(actor, document_id))

    @router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_document(document_id: str, actor: Actor) -> Response:
        await knowledge_service.delete(actor, document_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.get("/{document_id}/acl", response_model=DocumentAclResponse)
    def get_document_acl(document_id: str, actor: Actor) -> DocumentAclResponse:
        return _acl_response(document_id, knowledge_service.get_acl(actor, document_id))

    @router.put("/{document_id}/acl", response_model=DocumentAclResponse)
    def replace_document_acl(
        document_id: str, body: DocumentAclReplaceRequest, actor: Actor
    ) -> DocumentAclResponse:
        grants = tuple(
            AclGrant(
                user_id=item.user_id,
                role_id=item.role_id,
                access_level=item.access_level,
            )
            for item in body.grants
        )
        return _acl_response(
            document_id,
            knowledge_service.replace_acl(actor, document_id, grants),
        )

    return router


def _document_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        title=document.title,
        created_by_user_id=document.created_by_user_id,
        original_filename=document.original_filename,
        media_type=document.media_type,
        byte_size=document.byte_size,
        content_hash=document.content_hash,
        extracted_characters=document.extracted_characters,
        chunk_count=document.chunk_count,
        index_state=document.index_state,
        index_error=document.index_error,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def _acl_response(
    document_id: str, grants: tuple[AclGrant, ...]
) -> DocumentAclResponse:
    return DocumentAclResponse(
        document_id=document_id,
        grants=[
            AclGrantResponse(
                user_id=grant.user_id,
                role_id=grant.role_id,
                access_level=grant.access_level,
            )
            for grant in grants
        ],
    )


async def _require_multipart_fields(
    request: Request, expected: dict[str, int]
) -> None:
    form = await request.form()
    counts: dict[str, int] = {}
    for name, _ in form.multi_items():
        counts[name] = counts.get(name, 0) + 1
    if counts != expected:
        raise ApiError(422, "invalid_request", "Multipart fields are invalid")
