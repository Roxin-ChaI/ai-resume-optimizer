"""Offline tests for the model-provider abstraction and DeepSeek adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from ai_resume_optimizer.config import DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_MODEL
from ai_resume_optimizer.exceptions import ModelCallError, ModelOutputError
from ai_resume_optimizer.model_client import DeepSeekModelClient, ModelClient
from ai_resume_optimizer.models import (
    JobProfile,
    JobRequirement,
    OptimizedResume,
    StructuredResume,
)
from tests.fakes import FakeModelClient


class _ManyErrorModel(BaseModel):
    field_1: int
    field_2: int
    field_3: int
    field_4: int
    field_5: int
    field_6: int


def _job_profile() -> JobProfile:
    return JobProfile(
        role_summary="Build APIs.",
        requirements=[
            JobRequirement(
                requirement_id="requirement-0001",
                category="core_skill",
                description="Python",
                importance="required",
                source_excerpt="Python",
            )
        ],
    )


def _response(content: str | None, *, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ]
    )


def _injected_client(response: object) -> Mock:
    client = Mock()
    client.chat.completions.create.return_value = response
    return client


def _client(injected: Mock) -> DeepSeekModelClient:
    return DeepSeekModelClient(
        api_key="invalid-test-key",
        model=DEFAULT_DEEPSEEK_MODEL,
        timeout_seconds=12,
        client=injected,
    )


def test_fake_model_client_matches_protocol_call_shape() -> None:
    expected = _job_profile()
    client: ModelClient = FakeModelClient({JobProfile: expected})

    actual = client.generate_structured(
        instructions="Extract.",
        input_text="Python",
        response_model=JobProfile,
    )

    assert actual is expected


def test_constructor_creates_sdk_client_with_fixed_deepseek_settings() -> None:
    with patch("ai_resume_optimizer.model_client.OpenAI") as sdk_constructor:
        DeepSeekModelClient(
            api_key="  invalid-test-key  ",
            model=DEFAULT_DEEPSEEK_MODEL,
            timeout_seconds=12,
        )

    sdk_constructor.assert_called_once_with(
        api_key="invalid-test-key",
        base_url=DEEPSEEK_BASE_URL,
        timeout=12.0,
        max_retries=0,
    )


def test_constructor_uses_injected_client_without_creating_sdk_client() -> None:
    injected = Mock()
    with patch("ai_resume_optimizer.model_client.OpenAI") as sdk_constructor:
        client = DeepSeekModelClient(
            api_key="invalid-test-key",
            model=DEFAULT_DEEPSEEK_MODEL,
            timeout_seconds=12,
            client=injected,
        )

    sdk_constructor.assert_not_called()
    assert client._client is injected


def test_close_closes_injected_sdk_client_at_most_once() -> None:
    injected = Mock()
    client = _client(injected)

    client.close()
    client.close()

    injected.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_key", ""),
        ("api_key", "  "),
        ("model", ""),
        ("model", "deepseek-v4-pro"),
        ("model", "deepseek-chat"),
        ("model", "deepseek-reasoner"),
        ("timeout_seconds", 0),
        ("timeout_seconds", -1),
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", float("inf")),
    ],
)
def test_constructor_rejects_invalid_arguments(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "api_key": "invalid-test-key",
        "model": DEFAULT_DEEPSEEK_MODEL,
        "timeout_seconds": 12,
        "client": Mock(),
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        DeepSeekModelClient(**arguments)  # type: ignore[arg-type]


def test_constructor_does_not_read_environment_or_access_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_getenv(name: str) -> str:
        raise AssertionError(f"Constructor must not read {name}.")

    monkeypatch.setattr("ai_resume_optimizer.config.os.getenv", fail_getenv)
    injected = Mock()

    DeepSeekModelClient(
        api_key="invalid-test-key",
        model=DEFAULT_DEEPSEEK_MODEL,
        timeout_seconds=12,
        client=injected,
    )

    injected.assert_not_called()


def test_generate_structured_uses_chat_json_output_and_local_validation() -> None:
    expected = _job_profile()
    original_input = "  Python role  "
    injected = _injected_client(_response(expected.model_dump_json()))

    actual = _client(injected).generate_structured(
        instructions="  Extract requirements.  ",
        input_text=original_input,
        response_model=JobProfile,
    )

    assert actual == expected
    assert actual is not expected
    injected.chat.completions.create.assert_called_once()
    call_arguments = injected.chat.completions.create.call_args.kwargs
    assert call_arguments == {
        "model": DEFAULT_DEEPSEEK_MODEL,
        "messages": [
            {
                "role": "system",
                "content": call_arguments["messages"][0]["content"],
            },
            {"role": "user", "content": original_input},
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    system_content = call_arguments["messages"][0]["content"]
    assert system_content.startswith("Extract requirements.")
    assert "JSON" in system_content
    assert "exactly one JSON object" in system_content
    assert "Do not output Markdown or code fences" in system_content
    schema_text = system_content.split("JSON Schema:\n", maxsplit=1)[1]
    assert json.loads(schema_text) == JobProfile.model_json_schema()
    assert schema_text == json.dumps(
        JobProfile.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
    )
    for forbidden in (
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning_effort",
        "store",
        "background",
    ):
        assert forbidden not in call_arguments
    assert not hasattr(injected, "responses") or not injected.responses.called


def test_generate_structured_normalizes_optimized_resume_section_source_ids() -> None:
    content = json.dumps(
        {
            "sections": [
                {
                    "section_type": "experience",
                    "title": "Experience",
                    "items": [
                        {
                            "text": "Example experience",
                            "source_block_ids": ["block-0003", "block-0004"],
                            "related_requirement_ids": ["requirement-0001"],
                            "needs_review": False,
                            "review_note": None,
                        },
                        {
                            "text": "Example result",
                            "source_block_ids": ["block-0004", "block-0001"],
                            "related_requirement_ids": [],
                            "needs_review": False,
                            "review_note": None,
                        },
                    ],
                    "source_block_ids": ["block-9999"],
                },
                {
                    "section_type": "projects",
                    "title": "Projects",
                    "items": [
                        {
                            "text": "Example project",
                            "source_block_ids": ["block-0006", "block-0005"],
                            "related_requirement_ids": [],
                            "needs_review": True,
                            "review_note": "Review this fictitious wording.",
                        }
                    ],
                    "source_block_ids": ["block-0005", "block-0006"],
                },
            ],
            "pending_user_inputs": [],
            "warnings": [],
        }
    )
    injected = _injected_client(_response(content))

    actual = _client(injected).generate_structured(
        instructions="Optimize.",
        input_text="Fictitious resume input",
        response_model=OptimizedResume,
    )

    assert [section.source_block_ids for section in actual.sections] == [
        ["block-0003", "block-0004", "block-0001"],
        ["block-0006", "block-0005"],
    ]
    assert [item.text for item in actual.sections[0].items] == [
        "Example experience",
        "Example result",
    ]
    assert [item.source_block_ids for item in actual.sections[0].items] == [
        ["block-0003", "block-0004"],
        ["block-0004", "block-0001"],
    ]
    assert actual.sections[1].items[0].text == "Example project"
    assert actual.sections[1].items[0].source_block_ids == [
        "block-0006",
        "block-0005",
    ]
    injected.chat.completions.create.assert_called_once()


def test_generate_structured_normalizes_structured_resume_section_source_ids() -> None:
    content = json.dumps(
        {
            "sections": [
                {
                    "section_type": "skills",
                    "title": "Skills",
                    "items": [
                        {
                            "text": "Example skill",
                            "source_block_ids": ["block-0002", "block-0001"],
                            "related_requirement_ids": [],
                            "needs_review": False,
                            "review_note": None,
                        }
                    ],
                    "source_block_ids": ["block-0001"],
                }
            ],
            "unclassified_content": [],
            "warnings": [],
        }
    )
    injected = _injected_client(_response(content))

    actual = _client(injected).generate_structured(
        instructions="Structure.",
        input_text="Fictitious resume input",
        response_model=StructuredResume,
    )

    assert actual.sections[0].source_block_ids == ["block-0002", "block-0001"]
    assert actual.sections[0].items[0].source_block_ids == ["block-0002", "block-0001"]
    assert actual.sections[0].items[0].text == "Example skill"
    injected.chat.completions.create.assert_called_once()


@pytest.mark.parametrize(("instructions", "input_text"), [("", "x"), ("x", "  ")])
def test_generate_structured_rejects_blank_text(
    instructions: str,
    input_text: str,
) -> None:
    with pytest.raises(ValueError):
        _client(Mock()).generate_structured(
            instructions=instructions,
            input_text=input_text,
            response_model=JobProfile,
        )


@pytest.mark.parametrize(
    ("response", "expected_message"),
    [
        (SimpleNamespace(choices=[]), "choices"),
        (_response(None), "empty content"),
        (_response("   "), "empty content"),
        (_response("{}", finish_reason="length"), "truncated"),
        (
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=None)]),
            "message",
        ),
    ],
)
def test_generate_structured_rejects_missing_or_truncated_output(
    response: object,
    expected_message: str,
) -> None:
    with pytest.raises(ModelOutputError, match=expected_message) as raised:
        _client(_injected_client(response)).generate_structured(
            instructions="Extract.",
            input_text="Python",
            response_model=JobProfile,
        )

    assert "JobProfile" in str(raised.value)


def test_generate_structured_rejects_invalid_json_with_cause() -> None:
    sensitive_value = "TOP-SECRET-TEST-VALUE"
    invalid_json = f'{{\n"role_summary": "{sensitive_value}",\n"requirements": [}}'

    with pytest.raises(ModelOutputError) as raised:
        _client(_injected_client(_response(invalid_json))).generate_structured(
            instructions="Extract.",
            input_text="Fictitious input",
            response_model=JobProfile,
        )

    message = str(raised.value)
    assert isinstance(raised.value.__cause__, json.JSONDecodeError)
    assert "JobProfile JSON decoding failed" in message
    assert "line 3" in message
    assert "column" in message
    assert sensitive_value not in message
    assert invalid_json not in message
    assert "input_value" not in message
    assert "invalid-test-key" not in message


def test_generate_structured_rejects_non_object_json() -> None:
    sensitive_value = "TOP-SECRET-TEST-VALUE"
    content = json.dumps([{"secret": sensitive_value}])

    with pytest.raises(ModelOutputError) as raised:
        _client(_injected_client(_response(content))).generate_structured(
            instructions="Extract.",
            input_text="Python",
            response_model=JobProfile,
        )

    message = str(raised.value)
    assert "JobProfile output validation failed" in message
    assert "expected a JSON object" in message
    assert "received list" in message
    assert sensitive_value not in message
    assert content not in message


def test_generate_structured_reports_safe_pydantic_field_error_with_cause() -> None:
    sensitive_value = "TOP-SECRET-TEST-VALUE"
    content = json.dumps(
        {
            "role_summary": "Example role",
            "requirements": [
                {
                    "requirement_id": "requirement-0001",
                    "category": sensitive_value,
                    "description": "Example requirement",
                    "importance": "required",
                    "source_excerpt": "Example requirement",
                }
            ],
        }
    )

    with pytest.raises(ModelOutputError) as raised:
        _client(_injected_client(_response(content))).generate_structured(
            instructions="Extract.",
            input_text="Python",
            response_model=JobProfile,
        )

    message = str(raised.value)
    assert isinstance(raised.value.__cause__, ValidationError)
    assert "JobProfile output validation failed" in message
    assert "requirements.0.category [literal_error]" in message
    assert "Input should be" in message
    for forbidden in (
        sensitive_value,
        content,
        "input_value",
        "input_type",
        "errors.pydantic.dev",
        "invalid-test-key",
    ):
        assert forbidden not in message


def test_generate_structured_reports_safe_nested_validation_path() -> None:
    sensitive_value = "TOP-SECRET-TEST-VALUE"
    content = json.dumps(
        {
            "sections": [
                {
                    "section_type": "skills",
                    "title": "Example section",
                    "items": [
                        {
                            "text": "Example item",
                            "source_block_ids": sensitive_value,
                            "related_requirement_ids": [],
                            "needs_review": False,
                            "review_note": None,
                        }
                    ],
                    "source_block_ids": ["block-0001"],
                }
            ],
            "unclassified_content": [],
            "warnings": [],
        }
    )

    with pytest.raises(ModelOutputError) as raised:
        _client(_injected_client(_response(content))).generate_structured(
            instructions="Structure.",
            input_text="Fictitious resume",
            response_model=StructuredResume,
        )

    message = str(raised.value)
    assert "StructuredResume output validation failed" in message
    assert "sections.0.items.0.source_block_ids [list_type]" in message
    assert sensitive_value not in message
    assert content not in message


def test_generate_structured_limits_pydantic_diagnostics_to_five_issues() -> None:
    sensitive_value = "TOP-SECRET-TEST-VALUE"
    content = json.dumps(
        {
            "field_1": "one",
            "field_2": "two",
            "field_3": "three",
            "field_4": "four",
            "field_5": "five",
            "field_6": sensitive_value,
        }
    )

    with pytest.raises(ModelOutputError) as raised:
        _client(_injected_client(_response(content))).generate_structured(
            instructions="Validate.",
            input_text="Fictitious input",
            response_model=_ManyErrorModel,
        )

    message = str(raised.value)
    assert "_ManyErrorModel output validation failed" in message
    assert message.count("[int_parsing]") == 5
    assert "(+1 more validation errors)" in message
    assert "field_6" not in message
    assert sensitive_value not in message
    assert content not in message


def _api_errors() -> list[Exception]:
    request = httpx.Request("POST", f"{DEEPSEEK_BASE_URL}/chat/completions")
    unauthorized = httpx.Response(401, request=request)
    rate_limited = httpx.Response(429, request=request)
    server_error = httpx.Response(500, request=request)
    return [
        AuthenticationError("auth detail", response=unauthorized, body=None),
        APITimeoutError(request=request),
        APIConnectionError(message="connection detail", request=request),
        RateLimitError("rate detail", response=rate_limited, body=None),
        APIStatusError("status detail", response=server_error, body=None),
    ]


@pytest.mark.parametrize("provider_error", _api_errors())
def test_generate_structured_translates_public_sdk_errors(
    provider_error: Exception,
) -> None:
    injected = Mock()
    injected.chat.completions.create.side_effect = provider_error
    client = _client(injected)
    sensitive_input = "private resume text that must not leak"

    with pytest.raises(ModelCallError) as raised:
        client.generate_structured(
            instructions="Extract.",
            input_text=sensitive_input,
            response_model=JobProfile,
        )

    assert raised.value.__cause__ is provider_error
    assert "invalid-test-key" not in str(raised.value)
    assert sensitive_input not in str(raised.value)
    assert str(provider_error) not in str(raised.value)
