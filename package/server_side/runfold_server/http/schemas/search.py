from __future__ import annotations

from typing import Annotated

from pydantic import Field

from runfold_server.http.schemas.auth import StrictModel


class SearchRequest(StrictModel):
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(ge=1, le=100)
    document_ids: list[Annotated[str, Field(max_length=128)]] | None = Field(
        default=None, max_length=1_000
    )


class SearchResultResponse(StrictModel):
    document_id: str
    title: str
    ordinal: int
    content_hash: str
    text: str
    distance: float


class SearchResponse(StrictModel):
    items: list[SearchResultResponse]
