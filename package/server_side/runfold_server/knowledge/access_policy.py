from __future__ import annotations

import sqlite3

from runfold_server.access_control.audit import AuditRepository
from runfold_server.access_control.models import CurrentAccess
from runfold_server.errors import ApiError
from runfold_server.identity.models import AuthContext
from runfold_server.knowledge.models import Document
from runfold_server.knowledge.repository import KnowledgeRepository


class KnowledgeAccessPolicy:
    def __init__(
        self, repository: KnowledgeRepository, audit: AuditRepository
    ) -> None:
        self._repository = repository
        self._audit = audit

    def require_document(
        self,
        connection: sqlite3.Connection,
        *,
        context: AuthContext,
        access: CurrentAccess,
        document_id: str,
        minimum_level: int,
        states: tuple[str, ...],
        now: str,
    ) -> Document:
        if access.bypass:
            document = self._repository.get(connection, document_id)
            if document is None or document.index_state not in states:
                raise _not_found()
            self.record_bypass(
                connection,
                context=context,
                access=access,
                document_id=document_id,
                minimum_level=minimum_level,
                now=now,
            )
            return document
        document = self._repository.get_authorized(
            connection,
            access=access,
            document_id=document_id,
            minimum_level=minimum_level,
            states=states,
        )
        if document is None:
            raise _not_found()
        return document

    def record_bypass(
        self,
        connection: sqlite3.Connection,
        *,
        context: AuthContext,
        access: CurrentAccess,
        document_id: str | None,
        minimum_level: int,
        now: str,
    ) -> None:
        if not access.bypass:
            return
        self._audit.record(
            connection,
            actor_user_id=access.user_id,
            action="rag.document.bypass",
            decision="allowed",
            resource_type="document",
            resource_id=document_id,
            reason=None,
            request_id=context.request_id,
            details={"minimum_level": minimum_level},
            now=now,
        )


def _not_found() -> ApiError:
    return ApiError(404, "document_not_found", "Document not found")
