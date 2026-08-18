from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EffectiveLimits:
    max_documents: int
    max_storage_bytes: int
    monthly_embedding_tokens: int


@dataclass(frozen=True, slots=True)
class UsageSummary:
    document_count: int
    storage_bytes: int
    embedding_tokens: int
    limits: EffectiveLimits
