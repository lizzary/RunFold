from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EffectiveLimits:
    max_documents: int
    max_storage_bytes: int
    monthly_embedding_tokens: int


@dataclass(frozen=True, slots=True)
class LimitOverrides:
    max_documents: int | None
    max_storage_bytes: int | None
    monthly_embedding_tokens: int | None


@dataclass(frozen=True, slots=True)
class QuotaUsage:
    current: int
    limit: int
    remaining: int


@dataclass(frozen=True, slots=True)
class UsageSummary:
    user_id: str
    month_utc: str
    documents: QuotaUsage
    storage_bytes: QuotaUsage
    embedding_tokens: QuotaUsage
