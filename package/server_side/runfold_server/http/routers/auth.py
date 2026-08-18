from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, status

from runfold_server.http.auth_context import create_identity_dependency
from runfold_server.http.schemas.auth import (
    LoginRequest,
    LoginResponse,
    PasswordChangeRequest,
    UserResponse,
)
from runfold_server.identity.models import VerifiedIdentity
from runfold_server.identity.service import IdentityService


def create_auth_router(identity_service: IdentityService) -> APIRouter:
    router = APIRouter(prefix="/api/auth", tags=["authentication"])
    current_identity = create_identity_dependency(identity_service)

    @router.post("/login", response_model=LoginResponse)
    def login(body: LoginRequest, request: Request) -> LoginResponse:
        result = identity_service.login(body.username, body.password, request.state.request_id)
        request.state.actor_id = result.identity.user_id
        return LoginResponse(
            token=result.token,
            expires_at=result.expires_at,
            user=UserResponse.model_validate(result.identity.user),
        )

    @router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
    def logout(identity: Annotated[VerifiedIdentity, Depends(current_identity)]) -> Response:
        identity_service.logout(identity)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.get("/me", response_model=UserResponse)
    def me(identity: Annotated[VerifiedIdentity, Depends(current_identity)]) -> UserResponse:
        return UserResponse.model_validate(identity.user)

    @router.put("/password", status_code=status.HTTP_204_NO_CONTENT)
    def change_password(
        body: PasswordChangeRequest,
        identity: Annotated[VerifiedIdentity, Depends(current_identity)],
    ) -> Response:
        identity_service.change_password(
            identity, body.current_password, body.new_password
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
