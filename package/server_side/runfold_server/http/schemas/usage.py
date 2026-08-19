from __future__ import annotations

from pydantic import Field

from runfold_server.http.schemas.auth import StrictModel


class LimitsReplaceRequest(StrictModel):
    max_documents: int | None = Field(gt=0)
    max_storage_bytes: int | None = Field(gt=0)
    monthly_embedding_tokens: int | None = Field(gt=0)


class QuotaUsageResponse(StrictModel):
    current: int
    limit: int
    remaining: int


class UsageResponse(StrictModel):
    user_id: str
    month_utc: str
    documents: QuotaUsageResponse
    storage_bytes: QuotaUsageResponse
    embedding_tokens: QuotaUsageResponse
