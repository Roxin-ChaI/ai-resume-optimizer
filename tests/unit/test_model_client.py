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
from pydantic import ValidationError

from ai_resume_optimizer.config import DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_MODEL
from ai_resume_optimizer.exceptions import ModelCallError, ModelOutputError
from ai_resume_optimizer.model_client import DeepSeekModelClient, ModelClient
from ai_resume_optimizer.models import JobProfile, JobRequirement
from tests.fakes import FakeModelClient


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
        (_response(None), "JSON content"),
        (_response("   "), "JSON content"),
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
    with pytest.raises(ModelOutputError, match=expected_message):
        _client(_injected_client(response)).generate_structured(
            instructions="Extract.",
            input_text="Python",
            response_model=JobProfile,
        )


def test_generate_structured_rejects_invalid_json_with_cause() -> None:
    sensitive_input = "private resume text that must not leak"

    with pytest.raises(ModelOutputError) as raised:
        _client(_injected_client(_response("not-json"))).generate_structured(
            instructions="Extract.",
            input_text=sensitive_input,
            response_model=JobProfile,
        )

    assert isinstance(raised.value.__cause__, json.JSONDecodeError)
    assert sensitive_input not in str(raised.value)
    assert "not-json" not in str(raised.value)


def test_generate_structured_rejects_non_object_json() -> None:
    with pytest.raises(ModelOutputError, match="must be an object"):
        _client(_injected_client(_response("[]"))).generate_structured(
            instructions="Extract.",
            input_text="Python",
            response_model=JobProfile,
        )


def test_generate_structured_rejects_pydantic_validation_failure_with_cause() -> None:
    with pytest.raises(ModelOutputError) as raised:
        _client(
            _injected_client(_response('{"role_summary": "", "requirements": []}'))
        ).generate_structured(
            instructions="Extract.",
            input_text="Python",
            response_model=JobProfile,
        )

    assert isinstance(raised.value.__cause__, ValidationError)


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
