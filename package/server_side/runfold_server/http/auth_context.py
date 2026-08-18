from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from runfold_server.errors import ApiError
from runfold_server.identity.models import VerifiedIdentity
from runfold_server.identity.service import IdentityService

_BEARER = HTTPBearer(auto_error=False)


def create_identity_dependency(identity_service: IdentityService):
    def current_identity(
        request: Request,
        credentials: Annotated[
            HTTPAuthorizationCredentials | None, Depends(_BEARER)
        ] = None,
    ) -> VerifiedIdentity:
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise ApiError(401, "invalid_session", "Authentication is required")
        identity = identity_service.authenticate(
            credentials.credentials, request.state.request_id
        )
        request.state.actor_id = identity.user_id
        return identity

    return current_identity
