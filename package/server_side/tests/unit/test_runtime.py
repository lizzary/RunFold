from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from runfold_server.config import AgentBudget
from runfold_server.errors import ApiError
from runfold_server.identity.models import AuthContext, User, VerifiedIdentity
from runfold_server.runtime.files import FileWorkspaceService
from runfold_server.runtime.service import AgentRuntimeService
from runfold_server.runtime.skill_registry import SkillRegistry

_SKILL_ROOT = Path(__file__).parents[2] / "runfold_server" / "runtime" / "skills"


class _Identity:
    def revalidate(
        self, context: AuthContext, *, connection: object | None = None
    ) -> VerifiedIdentity:
        del connection
        return _actor(context.request_id)


class _Authorization:
    def require_capabilities(
        self,
        user_id: str,
        required: frozenset[str],
        *,
        connection: object | None = None,
    ) -> object:
        del user_id, required, connection
        return object()


class _Audit:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def record(self, connection: object, **values: object) -> None:
        del connection
        self.events.append(values)


class _Knowledge:
    async def search(self, *args: object, **kwargs: object) -> tuple[()]:
        del args, kwargs
        return ()


class _Usage:
    def __init__(self) -> None:
        self.tokens = 0

    def require_agent_capacity(self, connection: object, *, user_id: str) -> None:
        del connection, user_id

    def record_agent_tokens(self, user_id: str, tokens: int) -> None:
        del user_id
        self.tokens += tokens


class _QuotaUsage(_Usage):
    def require_agent_capacity(self, connection: object, *, user_id: str) -> None:
        del connection, user_id
        raise ApiError(
            429,
            "quota_exceeded",
            "Usage quota exceeded",
            details={"quota": "agent_tokens"},
        )


class _TeamModel(BaseChatModel):
    seen_prompts: ClassVar[list[str]] = []
    seen_tool_sets: ClassVar[list[set[str]]] = []
    seen_reasoning_efforts: ClassVar[list[str | None]] = []

    @property
    def _llm_type(self) -> str:
        return "runfold-scripted-team"

    def bind_tools(self, tools: list[Any], **kwargs: object) -> _TeamModel:
        self.seen_reasoning_efforts.append(getattr(self, "reasoning_effort", None))
        self.seen_tool_sets.append(
            {
                str(tool.name if hasattr(tool, "name") else tool.get("name", ""))
                for tool in tools
            }
        )
        del kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        system = next(
            str(message.content)
            for message in messages
            if isinstance(message, SystemMessage)
        )
        self.seen_prompts.append(system)
        tool_names = [message.name for message in messages if isinstance(message, ToolMessage)]
        human_count = sum(isinstance(message, HumanMessage) for message in messages)

        if "You are /root," in system:
            if "delegate_tasks" not in tool_names:
                answer = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "delegate_tasks",
                            "args": {
                                "tasks": [
                                    {
                                        "name": "lead",
                                        "task": "/product-decision synthesize the options",
                                    },
                                    {
                                        "name": "researcher",
                                        "task": "/rag-research gather the facts",
                                    },
                                ]
                            },
                            "id": "root-delegate",
                            "type": "tool_call",
                        }
                    ],
                )
            elif "message_agent" not in tool_names:
                answer = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "message_agent",
                            "args": {
                                "agent_path": "/root/researcher",
                                "message": "Clarify the evidence quality.",
                            },
                            "id": "root-follow-up",
                            "type": "tool_call",
                        }
                    ],
                )
            else:
                answer = AIMessage(
                    content="Root evaluated the reports and approved the result.",
                    additional_kwargs={
                        "reasoning_content": "Root compared every available employee report."
                    },
                )
        elif "You are /root/lead," in system:
            if "delegate_tasks" not in tool_names:
                answer = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "delegate_tasks",
                            "args": {
                                "tasks": [
                                    {
                                        "name": "reviewer",
                                        "task": "/critical-review review the proposal",
                                    }
                                ]
                            },
                            "id": "lead-delegate",
                            "type": "tool_call",
                        }
                    ],
                )
            else:
                answer = AIMessage(content="Lead assessed the reviewer and recommends approval.")
        elif "You are /root/lead/reviewer," in system:
            answer = AIMessage(content="Reviewer found no material unsupported claim.")
        elif "You are /root/researcher," in system:
            suffix = "clarified" if human_count > 1 else "initial"
            answer = AIMessage(content=f"Researcher returned {suffix} authorized evidence.")
        else:
            raise AssertionError(system)
        answer.usage_metadata = {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        }
        return ChatResult(generations=[ChatGeneration(message=answer)])


class _ConcurrencyModel(BaseChatModel):
    active: ClassVar[int] = 0
    peak: ClassVar[int] = 0
    seen_reasoning_efforts: ClassVar[list[str | None]] = []

    @property
    def _llm_type(self) -> str:
        return "runfold-concurrency-probe"

    def bind_tools(self, tools: object, **kwargs: object) -> _ConcurrencyModel:
        del tools, kwargs
        self.seen_reasoning_efforts.append(getattr(self, "reasoning_effort", None))
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        return _final_probe_result()

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        type(self).active += 1
        type(self).peak = max(type(self).peak, type(self).active)
        await asyncio.sleep(0.02)
        type(self).active -= 1
        return _final_probe_result()


class _UsageModel(BaseChatModel):
    usage: ClassVar[dict[str, Any]] = {}

    @property
    def _llm_type(self) -> str:
        return "runfold-usage-probe"

    def bind_tools(self, tools: object, **kwargs: object) -> _UsageModel:
        del tools, kwargs
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content="done", usage_metadata=self.usage)
                )
            ]
        )

def test_dynamic_tree_parallel_delegation_follow_up_and_skill_injection(
    tmp_path: Path,
) -> None:
    model = _TeamModel()
    model.seen_prompts.clear()
    model.seen_tool_sets.clear()
    model.seen_reasoning_efforts.clear()
    audit = _Audit()
    usage = _Usage()
    service = AgentRuntimeService(
        database_path=tmp_path / "runtime.sqlite3",
        identity=_Identity(),  # type: ignore[arg-type]
        authorization=_Authorization(),  # type: ignore[arg-type]
        audit=audit,  # type: ignore[arg-type]
        knowledge=_Knowledge(),  # type: ignore[arg-type]
        usage=usage,  # type: ignore[arg-type]
        model=model,
        skills=SkillRegistry(_SKILL_ROOT),
        budget=_budget(),
        provider_slots=asyncio.Semaphore(_budget().provider_concurrency),
        file_workspaces=_file_workspaces(tmp_path / "agent-work"),
        thinking_level_options=("on", "off", "low", "high"),
        default_thinking_level=None,
    )

    result = asyncio.run(service.run(_actor("request-1"), "Make the product decision."))

    assert result.answer == "Root evaluated the reports and approved the result."
    assert result.reasoning_content == "Root compared every available employee report."
    assert result.thinking_level is None
    assert result.agents_created == 3
    assert result.max_depth_reached == 2
    assert usage.tokens == 16
    assert any('<trusted_skill name="rag-research">' in item for item in model.seen_prompts)
    assert any('<trusted_skill name="critical-review">' in item for item in model.seen_prompts)
    assert any('<trusted_skill name="product-decision">' in item for item in model.seen_prompts)
    assert model.seen_tool_sets
    assert all(
        {
            "get_document_manifest",
            "read_document_text",
            "read_chunk_context",
            "search_document_text",
            "read_document_section",
            "write_file",
            "read_file",
            "read_files",
            "list_directory",
            "find_files",
            "search_files",
            "file_info",
            "count_text",
            "read_file_chunk",
            "append_file",
            "apply_patch",
        }.issubset(tool_set)
        for tool_set in model.seen_tool_sets
    )
    assert audit.events[-1]["action"] == "agent.run"
    assert audit.events[-1]["details"] == {
        "outcome": "completed",
        "agents_created": 3,
        "max_depth_reached": 2,
    }


def test_skill_registry_only_consumes_registered_slash_selectors() -> None:
    registry = SkillRegistry(_SKILL_ROOT)

    task, selected = registry.select_for_task(
        "/rag-research inspect /api/rag and /not-a-skill"
    )

    assert task == "inspect /api/rag and /not-a-skill"
    assert [skill.name for skill in selected] == ["rag-research"]
    assert {skill.name for skill in registry.catalog()} == {
        "critical-review",
        "product-decision",
        "rag-research",
    }


def test_agent_quota_is_checked_before_the_model_is_called(tmp_path: Path) -> None:
    model = _TeamModel()
    model.seen_prompts.clear()
    audit = _Audit()
    service = AgentRuntimeService(
        database_path=tmp_path / "runtime.sqlite3",
        identity=_Identity(),  # type: ignore[arg-type]
        authorization=_Authorization(),  # type: ignore[arg-type]
        audit=audit,  # type: ignore[arg-type]
        knowledge=_Knowledge(),  # type: ignore[arg-type]
        usage=_QuotaUsage(),  # type: ignore[arg-type]
        model=model,
        skills=SkillRegistry(_SKILL_ROOT),
        budget=_budget(),
        provider_slots=asyncio.Semaphore(_budget().provider_concurrency),
        file_workspaces=_file_workspaces(tmp_path / "agent-work"),
        thinking_level_options=("on", "off", "low", "high"),
        default_thinking_level=None,
    )

    with pytest.raises(ApiError) as captured:
        asyncio.run(service.run(_actor("request-quota"), "Do not call the model."))

    assert captured.value.status_code == 429
    assert captured.value.details == {"quota": "agent_tokens"}
    assert model.seen_prompts == []
    assert audit.events[-1]["decision"] == "denied"
    assert audit.events[-1]["reason"] == "quota_exceeded"


def test_provider_concurrency_is_enforced_across_root_runs(tmp_path: Path) -> None:
    async def exercise() -> tuple[int, int]:
        serialized_model = _ConcurrencyModel()
        _ConcurrencyModel.active = 0
        _ConcurrencyModel.peak = 0
        serialized = _runtime_service(
            tmp_path / "serialized.sqlite3",
            serialized_model,
            _budget(provider_concurrency=1),
        )
        await asyncio.gather(
            serialized.run(_actor("serialized-1"), "first"),
            serialized.run(_actor("serialized-2"), "second"),
        )
        serialized_peak = _ConcurrencyModel.peak

        parallel_model = _ConcurrencyModel()
        _ConcurrencyModel.active = 0
        _ConcurrencyModel.peak = 0
        parallel = _runtime_service(
            tmp_path / "parallel.sqlite3",
            parallel_model,
            _budget(provider_concurrency=2),
        )
        await asyncio.gather(
            parallel.run(_actor("parallel-1"), "first"),
            parallel.run(_actor("parallel-2"), "second"),
        )
        return serialized_peak, _ConcurrencyModel.peak

    assert asyncio.run(exercise()) == (1, 2)


@pytest.mark.parametrize(
    ("usage", "status_code", "code"),
    [
        (
            {"input_tokens": 5001, "output_tokens": 1, "total_tokens": 5002},
            413,
            "agent_input_token_limit",
        ),
        (
            {"input_tokens": 1, "output_tokens": 250, "total_tokens": 251},
            502,
            "agent_provider_budget_exceeded",
        ),
        (
            {
                "input_tokens": 1,
                "output_tokens": 400,
                "total_tokens": 401,
                "output_token_details": {"reasoning": 350},
            },
            502,
            "agent_provider_budget_exceeded",
        ),
    ],
)
def test_provider_usage_is_checked_against_each_configured_budget(
    tmp_path: Path,
    usage: dict[str, Any],
    status_code: int,
    code: str,
) -> None:
    _UsageModel.usage = usage
    service = _runtime_service(tmp_path / f"{code}.sqlite3", _UsageModel(), _budget())

    with pytest.raises(ApiError) as captured:
        asyncio.run(service.run(_actor(f"request-{status_code}"), "budget probe"))

    assert captured.value.status_code == status_code
    assert captured.value.code == code


def test_thinking_level_uses_config_default_and_allows_request_override(
    tmp_path: Path,
) -> None:
    model = _ConcurrencyModel()
    model.seen_reasoning_efforts.clear()
    service = _runtime_service(
        tmp_path / "thinking.sqlite3",
        model,
        _budget(),
        default_thinking_level="high",
    )

    configured_default = asyncio.run(
        service.run(_actor("thinking-default"), "default")
    )
    provider_default = asyncio.run(
        service.run(_actor("thinking-provider"), "provider", thinking_level="")
    )

    assert configured_default.thinking_level == "high"
    assert provider_default.thinking_level is None
    assert model.seen_reasoning_efforts == ["high", None]

    with pytest.raises(ApiError) as invalid:
        asyncio.run(
            service.run(
                _actor("thinking-invalid"),
                "invalid",
                thinking_level="medium",
            )
        )
    assert invalid.value.code == "invalid_thinking_level"


def _actor(request_id: str) -> VerifiedIdentity:
    return VerifiedIdentity(
        context=AuthContext(
            user_id="user-1",
            session_id="session-1",
            request_id=request_id,
        ),
        user=User(
            id="user-1",
            username="agent-user",
            display_name="Agent User",
            status="active",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        ),
    )


def _runtime_service(
    database_path: Path,
    model: BaseChatModel,
    budget: AgentBudget,
    *,
    default_thinking_level: str | None = None,
) -> AgentRuntimeService:
    return AgentRuntimeService(
        database_path=database_path,
        identity=_Identity(),  # type: ignore[arg-type]
        authorization=_Authorization(),  # type: ignore[arg-type]
        audit=_Audit(),  # type: ignore[arg-type]
        knowledge=_Knowledge(),  # type: ignore[arg-type]
        usage=_Usage(),  # type: ignore[arg-type]
        model=model,
        skills=SkillRegistry(_SKILL_ROOT),
        budget=budget,
        provider_slots=asyncio.Semaphore(budget.provider_concurrency),
        file_workspaces=_file_workspaces(database_path.parent / "agent-work"),
        thinking_level_options=("on", "off", "low", "high"),
        default_thinking_level=default_thinking_level,
    )


def _final_probe_result() -> ChatResult:
    return ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(
                    content="done",
                    usage_metadata={
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                    },
                )
            )
        ]
    )


def _file_workspaces(path: Path) -> FileWorkspaceService:
    path.mkdir(parents=True, exist_ok=True)
    return FileWorkspaceService(path)


def _budget(*, provider_concurrency: int = 2) -> AgentBudget:
    return AgentBudget(
        context_window_tokens=22000,
        provider_concurrency=provider_concurrency,
        input_tokens=5000,
        output_tokens=500,
        thinking_tokens=300,
        compression_threshold=0.8,
    )
