from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Capability:
    code: str
    description: str


@dataclass(frozen=True, slots=True)
class Role:
    id: str
    name: str
    description: str
    is_protected: bool
    capability_codes: tuple[str, ...]
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class CurrentAccess:
    user_id: str
    role_ids: frozenset[str]
    capabilities: frozenset[str]

    @property
    def bypass(self) -> bool:
        return "rag.document.bypass_acl" in self.capabilities


@dataclass(frozen=True, slots=True)
class AuditEvent:
    id: int
    actor_user_id: str | None
    action: str
    decision: str
    resource_type: str
    resource_id: str | None
    reason: str | None
    request_id: str
    details: dict[str, Any]
    created_at: str
