from __future__ import annotations

import json

import httpx2
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from runfold_server.llm.openai_chat import OpenAICompatibleChatModel


def test_openai_compatible_chat_preserves_reasoning_content() -> None:
    model = OpenAICompatibleChatModel(
        model="test-model",
        base_url="https://provider.example/v1",
        api_key="test-key",
        use_responses_api=False,
    )

    result = model._create_chat_result(
        {
            "id": "chatcmpl-test",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "visible answer",
                        "reasoning_content": "provider reasoning",
                        "tool_calls": [],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 5,
                "total_tokens": 8,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }
    )

    message = result.generations[0].message
    assert message.content == "visible answer"
    assert message.additional_kwargs["reasoning_content"] == "provider reasoning"
    assert message.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 5,
        "total_tokens": 8,
        "input_token_details": {},
        "output_token_details": {"reasoning": 2},
    }


def test_reasoning_is_replayed_after_tool_result_through_openai_sdk() -> None:
    requests: list[dict[str, object]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": f"chatcmpl-{len(requests)}",
                "object": "chat.completion",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "done",
                            "reasoning_content": "continued reasoning",
                            "tool_calls": [],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                    "completion_tokens_details": {"reasoning_tokens": 1},
                },
            },
        )

    with httpx2.Client(transport=httpx2.MockTransport(handle)) as http:
        model = OpenAICompatibleChatModel(
            model="test-model",
            base_url="https://provider.example/v1",
            api_key="test-key",
            use_responses_api=False,
            http_client=http,
        )
        model.invoke(
            [
                HumanMessage(content="Use a tool"),
                AIMessage(
                    content="",
                    additional_kwargs={
                        "reasoning_content": "reason before tool",
                        "reasoning_signature": "signature-before-tool",
                    },
                    tool_calls=[
                        {
                            "name": "probe",
                            "args": {},
                            "id": "call-probe",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    content="tool result",
                    tool_call_id="call-probe",
                    name="probe",
                ),
            ]
        )

    replayed = requests[0]["messages"][1]
    assert replayed["reasoning_content"] == "reason before tool"
    assert replayed["reasoning_signature"] == "signature-before-tool"
