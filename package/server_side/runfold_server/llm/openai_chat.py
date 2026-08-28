from __future__ import annotations

from typing import Any

import openai
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI


class OpenAICompatibleChatModel(ChatOpenAI):
    """Preserve reasoning_content returned by OpenAI-compatible providers."""

    def _create_chat_result(
        self,
        response: dict[str, Any] | openai.BaseModel,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        payload = response if isinstance(response, dict) else response.model_dump(warnings=False)
        choices = payload.get("choices") or ()
        for generation, choice in zip(result.generations, choices, strict=False):
            message = generation.message
            provider_message = choice.get("message") if isinstance(choice, dict) else None
            reasoning = (
                provider_message.get("reasoning_content")
                if isinstance(provider_message, dict)
                else None
            )
            if isinstance(message, AIMessage) and isinstance(reasoning, str):
                message.additional_kwargs["reasoning_content"] = reasoning
        return result

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        source_messages = self._convert_input(input_).to_messages()
        provider_messages = payload.get("messages") or ()
        for source, provider in zip(source_messages, provider_messages, strict=False):
            if not isinstance(source, AIMessage) or not isinstance(provider, dict):
                continue
            for field in ("reasoning_content", "reasoning_signature"):
                value = source.additional_kwargs.get(field)
                if value is not None:
                    provider[field] = value
        return payload
