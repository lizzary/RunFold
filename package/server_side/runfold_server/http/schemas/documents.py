from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from runfold_server.http.schemas.auth import StrictModel


class DocumentResponse(StrictModel):
    id: str
    title: str
    created_by_user_id: str
    original_filename: str
    media_type: str
    byte_size: int
    content_hash: str
    extracted_characters: int
    chunk_count: int
    index_state: Literal["ready", "failed"]
    index_error: str | None
    created_at: str
    updated_at: str


class DocumentsPage(StrictModel):
    items: list[DocumentResponse]
    limit: int
    offset: int
    total: int


class DocumentMetadataUpdateRequest(StrictModel):
    title: str = Field(min_length=1, max_length=200)


class DocumentTextReplaceRequest(StrictModel):
    text: str = Field(max_length=10_000_000)


class DocumentTextResponse(StrictModel):
    document_id: str
    text: str
    content_hash: str


class AclGrantRequest(StrictModel):
    user_id: str | None = None
    role_id: str | None = None
    access_level: Literal[10, 20, 30]

    @model_validator(mode="after")
    def exactly_one_subject(self) -> AclGrantRequest:
        if (self.user_id is None) == (self.role_id is None):
            raise ValueError("exactly one ACL subject is required")
        return self


class DocumentAclReplaceRequest(StrictModel):
    grants: list[AclGrantRequest] = Field(max_length=500)


class AclGrantResponse(StrictModel):
    user_id: str | None
    role_id: str | None
    access_level: Literal[10, 20, 30]


class DocumentAclResponse(StrictModel):
    document_id: str
    grants: list[AclGrantResponse]
