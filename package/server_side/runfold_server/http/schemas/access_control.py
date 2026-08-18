from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from runfold_server.http.schemas.auth import StrictModel, UserResponse


class UserCreateRequest(StrictModel):
    username: str = Field(min_length=1, max_length=64)
    display_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=256)


class UserUpdateRequest(StrictModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=100)
    status: Literal["active", "disabled"] | None = None

    @model_validator(mode="after")
    def require_field(self) -> UserUpdateRequest:
        if self.display_name is None and self.status is None:
            raise ValueError("at least one field is required")
        return self


class AdminPasswordResetRequest(StrictModel):
    new_password: str = Field(min_length=1, max_length=256)


class UserRolesReplaceRequest(StrictModel):
    role_ids: list[str] = Field(max_length=100)


class UserRolesResponse(StrictModel):
    user_id: str
    role_ids: list[str]


class UsersPage(StrictModel):
    items: list[UserResponse]
    limit: int
    offset: int
    total: int


class CapabilityResponse(StrictModel):
    code: str
    description: str


class CapabilitiesPage(StrictModel):
    items: list[CapabilityResponse]
    limit: int
    offset: int
    total: int


class RoleCreateRequest(StrictModel):
    name: str = Field(min_length=1, max_length=64)
    description: str = Field(default="", max_length=500)


class RoleUpdateRequest(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=64)
    description: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def require_field(self) -> RoleUpdateRequest:
        if self.name is None and self.description is None:
            raise ValueError("at least one field is required")
        return self


class RoleCapabilitiesReplaceRequest(StrictModel):
    capability_codes: list[str] = Field(max_length=100)


class RoleResponse(StrictModel):
    id: str
    name: str
    description: str
    is_protected: bool
    capability_codes: list[str]
    created_at: str
    updated_at: str


class RolesPage(StrictModel):
    items: list[RoleResponse]
    limit: int
    offset: int
    total: int
