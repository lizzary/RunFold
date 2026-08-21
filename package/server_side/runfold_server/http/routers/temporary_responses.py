import asyncio
from collections.abc import Mapping
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends
from pydantic import Field

from runfold_server.errors import ApiError
from runfold_server.http.auth_context import create_identity_dependency
from runfold_server.http.schemas.auth import StrictModel
from runfold_server.identity.models import VerifiedIdentity
from runfold_server.identity.service import IdentityService

_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


class TemporaryResponseRequest(StrictModel):
    model: str = Field(
        min_length=1,
        max_length=200,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$",
    )
    input: str = Field(min_length=1, max_length=100_000)
    previous_response_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        pattern=r"^\S+$",
    )


class TemporaryResponsesClient:
    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        max_retries: int,
    ) -> None:
        self._http = http_client
        self._endpoint = f"{base_url}/responses"
        self._api_key = api_key
        self._max_retries = max_retries

    async def create(
        self,
        *,
        model: str,
        input_text: str,
        previous_response_id: str | None,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        body: dict[str, Any] = {
            "model": model,
            "input": input_text,
            "store": True,
            "tools": [],
        }
        if previous_response_id is not None:
            body["previous_response_id"] = previous_response_id

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._http.post(
                    self._endpoint,
                    headers=headers,
                    json=body,
                )
            except httpx.RequestError:
                if attempt == self._max_retries:
                    raise _provider_error() from None
                await asyncio.sleep(0.1 * (2**attempt))
                continue
            if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                await asyncio.sleep(0.1 * (2**attempt))
                continue
            if response.status_code < 200 or response.status_code >= 300:
                raise _provider_error()
            return _validated_response(response)
        raise _provider_error()


def create_temporary_responses_router(
    identity_service: IdentityService,
    responses: TemporaryResponsesClient,
) -> APIRouter:
    router = APIRouter(prefix="/api/temporary", tags=["temporary-pm-testing"])
    current_identity = create_identity_dependency(identity_service)
    Actor = Annotated[VerifiedIdentity, Depends(current_identity)]

    @router.post("/responses", response_model=None)
    async def create_response(
        body: TemporaryResponseRequest,
        actor: Actor,
    ) -> dict[str, Any]:
        result = await responses.create(
            model=body.model,
            input_text=body.input,
            previous_response_id=body.previous_response_id,
        )
        identity_service.revalidate(actor.context)
        return result

    return router


def _validated_response(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        raise ApiError(
            502,
            "invalid_response_provider_response",
            "Response provider returned an invalid response",
        ) from None
    if not isinstance(payload, Mapping) or payload.get("object") != "response":
        raise ApiError(
            502,
            "invalid_response_provider_response",
            "Response provider returned an invalid response",
        )
    response_id = payload.get("id")
    if not isinstance(response_id, str) or not response_id:
        raise ApiError(
            502,
            "invalid_response_provider_response",
            "Response provider returned an invalid response",
        )
    return dict(payload)


def _provider_error() -> ApiError:
    return ApiError(
        502,
        "response_provider_error",
        "Response provider request failed",
    )
