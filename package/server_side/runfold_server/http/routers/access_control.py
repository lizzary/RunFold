from typing import Annotated

from fastapi import APIRouter, Depends, Query, Response, status

from runfold_server.access_control.models import Capability, Role
from runfold_server.access_control.service import AccessControlService
from runfold_server.http.auth_context import create_identity_dependency
from runfold_server.http.schemas.access_control import (
    AdminPasswordResetRequest,
    CapabilitiesPage,
    CapabilityResponse,
    RoleCapabilitiesReplaceRequest,
    RoleCreateRequest,
    RoleResponse,
    RolesPage,
    RoleUpdateRequest,
    UserCreateRequest,
    UserRolesReplaceRequest,
    UserRolesResponse,
    UsersPage,
    UserUpdateRequest,
)
from runfold_server.http.schemas.auth import UserResponse
from runfold_server.identity.models import VerifiedIdentity
from runfold_server.identity.service import IdentityService


def create_access_control_router(
    identity_service: IdentityService,
    access_control_service: AccessControlService,
) -> APIRouter:
    router = APIRouter(prefix="/api/access", tags=["access-control"])
    current_identity = create_identity_dependency(identity_service)
    Actor = Annotated[VerifiedIdentity, Depends(current_identity)]
    Limit = Annotated[int, Query(ge=1, le=100)]
    Offset = Annotated[int, Query(ge=0)]

    @router.get("/capabilities", response_model=CapabilitiesPage)
    def list_capabilities(actor: Actor, limit: Limit = 50, offset: Offset = 0) -> CapabilitiesPage:
        items, total = access_control_service.list_capabilities(
            actor, limit=limit, offset=offset
        )
        return CapabilitiesPage(
            items=[_capability_response(item) for item in items],
            limit=limit,
            offset=offset,
            total=total,
        )

    @router.get("/users", response_model=UsersPage)
    def list_users(actor: Actor, limit: Limit = 50, offset: Offset = 0) -> UsersPage:
        items, total = access_control_service.list_users(actor, limit=limit, offset=offset)
        return UsersPage(
            items=[UserResponse.model_validate(item) for item in items],
            limit=limit,
            offset=offset,
            total=total,
        )

    @router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
    def create_user(body: UserCreateRequest, actor: Actor) -> UserResponse:
        user = access_control_service.create_user(
            actor,
            username=body.username,
            display_name=body.display_name,
            password=body.password,
        )
        return UserResponse.model_validate(user)

    @router.get("/users/{user_id}", response_model=UserResponse)
    def get_user(user_id: str, actor: Actor) -> UserResponse:
        return UserResponse.model_validate(access_control_service.get_user(actor, user_id))

    @router.patch("/users/{user_id}", response_model=UserResponse)
    def update_user(user_id: str, body: UserUpdateRequest, actor: Actor) -> UserResponse:
        return UserResponse.model_validate(
            access_control_service.update_user(
                actor,
                user_id,
                display_name=body.display_name,
                status=body.status,
            )
        )

    @router.put("/users/{user_id}/password", status_code=status.HTTP_204_NO_CONTENT)
    def reset_password(
        user_id: str, body: AdminPasswordResetRequest, actor: Actor
    ) -> Response:
        access_control_service.reset_password(actor, user_id, body.new_password)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.put("/users/{user_id}/roles", response_model=UserRolesResponse)
    def replace_user_roles(
        user_id: str, body: UserRolesReplaceRequest, actor: Actor
    ) -> UserRolesResponse:
        role_ids = access_control_service.replace_user_roles(actor, user_id, body.role_ids)
        return UserRolesResponse(user_id=user_id, role_ids=list(role_ids))

    @router.get("/users/{user_id}/roles", response_model=UserRolesResponse)
    def get_user_roles(user_id: str, actor: Actor) -> UserRolesResponse:
        role_ids = access_control_service.get_user_roles(actor, user_id)
        return UserRolesResponse(user_id=user_id, role_ids=list(role_ids))

    @router.get("/roles", response_model=RolesPage)
    def list_roles(actor: Actor, limit: Limit = 50, offset: Offset = 0) -> RolesPage:
        items, total = access_control_service.list_roles(actor, limit=limit, offset=offset)
        return RolesPage(
            items=[_role_response(item) for item in items],
            limit=limit,
            offset=offset,
            total=total,
        )

    @router.post("/roles", response_model=RoleResponse, status_code=status.HTTP_201_CREATED)
    def create_role(body: RoleCreateRequest, actor: Actor) -> RoleResponse:
        role = access_control_service.create_role(
            actor, name=body.name, description=body.description
        )
        return _role_response(role)

    @router.get("/roles/{role_id}", response_model=RoleResponse)
    def get_role(role_id: str, actor: Actor) -> RoleResponse:
        return _role_response(access_control_service.get_role(actor, role_id))

    @router.patch("/roles/{role_id}", response_model=RoleResponse)
    def update_role(
        role_id: str, body: RoleUpdateRequest, actor: Actor
    ) -> RoleResponse:
        return _role_response(
            access_control_service.update_role(
                actor,
                role_id,
                name=body.name,
                description=body.description,
            )
        )

    @router.delete("/roles/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_role(role_id: str, actor: Actor) -> Response:
        access_control_service.delete_role(actor, role_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.put("/roles/{role_id}/capabilities", response_model=RoleResponse)
    def replace_role_capabilities(
        role_id: str,
        body: RoleCapabilitiesReplaceRequest,
        actor: Actor,
    ) -> RoleResponse:
        return _role_response(
            access_control_service.replace_role_capabilities(
                actor, role_id, body.capability_codes
            )
        )

    return router


def _capability_response(capability: Capability) -> CapabilityResponse:
    return CapabilityResponse(code=capability.code, description=capability.description)


def _role_response(role: Role) -> RoleResponse:
    return RoleResponse(
        id=role.id,
        name=role.name,
        description=role.description,
        is_protected=role.is_protected,
        capability_codes=list(role.capability_codes),
        created_at=role.created_at,
        updated_at=role.updated_at,
    )
