"""Offline tests for the model-provider abstraction and OpenAI adapter."""

from __future__ import annotations

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

from ai_resume_optimizer.exceptions import ModelCallError, ModelOutputError
from ai_resume_optimizer.model_client import ModelClient, OpenAIModelClient
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


def _injected_client(output: object) -> Mock:
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(output_parsed=output)
    return client


def test_fake_model_client_matches_protocol_call_shape() -> None:
    expected = _job_profile()
    client: ModelClient = FakeModelClient({JobProfile: expected})

    actual = client.generate_structured(
        instructions="Extract.",
        input_text="Python",
        response_model=JobProfile,
    )

    assert actual is expected


def test_constructor_creates_openai_client_with_explicit_safe_settings() -> None:
    with patch("ai_resume_optimizer.model_client.OpenAI") as openai_constructor:
        OpenAIModelClient(
            api_key="  test-key  ",
            model="  test-model  ",
            timeout_seconds=12,
        )

    openai_constructor.assert_called_once_with(
        api_key="test-key",
        timeout=12.0,
        max_retries=0,
    )


def test_constructor_uses_injected_client_without_creating_openai_client() -> None:
    injected = Mock()
    with patch("ai_resume_optimizer.model_client.OpenAI") as openai_constructor:
        OpenAIModelClient(
            api_key="test-key",
            model="test-model",
            timeout_seconds=12,
            client=injected,
        )

    openai_constructor.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_key", ""),
        ("api_key", "  "),
        ("model", ""),
        ("model", "  "),
        ("timeout_seconds", 0),
        ("timeout_seconds", -1),
        ("timeout_seconds", float("nan")),
        ("timeout_seconds", float("inf")),
    ],
)
def test_constructor_rejects_invalid_arguments(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "api_key": "test-key",
        "model": "test-model",
        "timeout_seconds": 12,
        "client": Mock(),
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        OpenAIModelClient(**arguments)  # type: ignore[arg-type]


def test_generate_structured_uses_responses_parse_and_revalidates() -> None:
    parsed = _job_profile()
    injected = _injected_client(parsed)
    client = OpenAIModelClient(
        api_key="test-key",
        model="test-model",
        timeout_seconds=12,
        client=injected,
    )

    actual = client.generate_structured(
        instructions="  Extract requirements.  ",
        input_text="  Python  ",
        response_model=JobProfile,
    )

    assert actual == parsed
    assert actual is not parsed
    injected.responses.parse.assert_called_once_with(
        model="test-model",
        instructions="Extract requirements.",
        input="Python",
        text_format=JobProfile,
        store=False,
    )
    call_arguments = injected.responses.parse.call_args.kwargs
    assert "tools" not in call_arguments
    assert "stream" not in call_arguments
    assert "background" not in call_arguments


@pytest.mark.parametrize(("instructions", "input_text"), [("", "x"), ("x", "  ")])
def test_generate_structured_rejects_blank_text(
    instructions: str,
    input_text: str,
) -> None:
    client = OpenAIModelClient(
        api_key="test-key",
        model="test-model",
        timeout_seconds=12,
        client=Mock(),
    )

    with pytest.raises(ValueError):
        client.generate_structured(
            instructions=instructions,
            input_text=input_text,
            response_model=JobProfile,
        )


@pytest.mark.parametrize(
    "output",
    [
        None,
        JobRequirement(
            requirement_id="requirement-0001",
            category="core_skill",
            description="Python",
            importance="required",
            source_excerpt="Python",
        ),
    ],
)
def test_generate_structured_rejects_missing_or_wrong_output(output: object) -> None:
    client = OpenAIModelClient(
        api_key="test-key",
        model="test-model",
        timeout_seconds=12,
        client=_injected_client(output),
    )

    with pytest.raises(ModelOutputError):
        client.generate_structured(
            instructions="Extract.",
            input_text="Python",
            response_model=JobProfile,
        )


def test_generate_structured_revalidates_constructed_target_model() -> None:
    invalid = JobProfile.model_construct(role_summary="", requirements=[])
    client = OpenAIModelClient(
        api_key="test-key",
        model="test-model",
        timeout_seconds=12,
        client=_injected_client(invalid),
    )

    with pytest.raises(ModelOutputError) as raised:
        client.generate_structured(
            instructions="Extract.",
            input_text="Python",
            response_model=JobProfile,
        )

    assert isinstance(raised.value.__cause__, ValidationError)


def test_generate_structured_rejects_response_without_public_parsed_output() -> None:
    injected = Mock()
    injected.responses.parse.return_value = object()
    client = OpenAIModelClient(
        api_key="test-key",
        model="test-model",
        timeout_seconds=12,
        client=injected,
    )

    with pytest.raises(ModelOutputError) as raised:
        client.generate_structured(
            instructions="Extract.",
            input_text="Python",
            response_model=JobProfile,
        )

    assert isinstance(raised.value.__cause__, AttributeError)


def _api_errors() -> list[Exception]:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
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
def test_generate_structured_translates_public_openai_errors(
    provider_error: Exception,
) -> None:
    injected = Mock()
    injected.responses.parse.side_effect = provider_error
    client = OpenAIModelClient(
        api_key="secret-test-key",
        model="test-model",
        timeout_seconds=12,
        client=injected,
    )
    sensitive_input = "private resume text that must not leak"

    with pytest.raises(ModelCallError) as raised:
        client.generate_structured(
            instructions="Extract.",
            input_text=sensitive_input,
            response_model=JobProfile,
        )

    assert raised.value.__cause__ is provider_error
    assert "secret-test-key" not in str(raised.value)
    assert sensitive_input not in str(raised.value)
    assert str(provider_error) not in str(raised.value)
