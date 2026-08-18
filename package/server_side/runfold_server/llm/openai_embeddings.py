from __future__ import annotations

import asyncio
import hashlib
import math
from dataclasses import dataclass

import httpx

from runfold_server.errors import ApiError

_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    total_tokens: int


class OpenAIEmbeddingsClient:
    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int,
        max_retries: int,
    ) -> None:
        self._http = http_client
        self._endpoint = f"{base_url}/embeddings"
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._max_retries = max_retries

    async def embed(self, values: tuple[str, ...]) -> EmbeddingBatch:
        if not values:
            return EmbeddingBatch(vectors=(), total_tokens=0)
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._http.post(
                    self._endpoint,
                    headers=headers,
                    json={"model": self._model, "input": list(values)},
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
            return self._validated_response(response, len(values))
        raise _provider_error()

    def _validated_response(self, response: httpx.Response, expected: int) -> EmbeddingBatch:
        try:
            payload = response.json()
            data = payload["data"]
            usage = payload["usage"]
            total_tokens = usage["total_tokens"]
            if (
                not isinstance(data, list)
                or len(data) != expected
                or isinstance(total_tokens, bool)
                or not isinstance(total_tokens, int)
                or total_tokens < 0
            ):
                raise ValueError
            vectors: list[tuple[float, ...]] = []
            for expected_index, item in enumerate(data):
                if not isinstance(item, dict) or item.get("index") != expected_index:
                    raise ValueError
                vector = item.get("embedding")
                if not isinstance(vector, list) or len(vector) != self._dimensions:
                    raise ValueError
                converted = tuple(float(value) for value in vector)
                if any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in vector
                ) or not all(math.isfinite(value) for value in converted):
                    raise ValueError
                vectors.append(converted)
        except (KeyError, TypeError, ValueError):
            raise ApiError(
                502,
                "invalid_embedding_response",
                "Embedding provider returned an invalid response",
            ) from None
        return EmbeddingBatch(vectors=tuple(vectors), total_tokens=total_tokens)


def embedding_identity(base_url: str, model: str, dimensions: int) -> str:
    value = f"{base_url}\0{model}\0{dimensions}".encode()
    return hashlib.sha256(value).hexdigest()


def _provider_error() -> ApiError:
    return ApiError(502, "embedding_provider_error", "Embedding provider request failed")
