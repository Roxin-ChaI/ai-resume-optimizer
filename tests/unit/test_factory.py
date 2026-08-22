"""Offline tests for the public production optimizer composition."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import ai_resume_optimizer
import ai_resume_optimizer.factory as factory_module
from ai_resume_optimizer import (
    ConfigurationError,
    InputError,
    InputTooLargeError,
    MatchAnalysis,
    ModelCallError,
    ModelClient,
    ModelOutputError,
    OptimizationResult,
    OptimizedResume,
    OutputError,
    RequirementAssessment,
    ResumeExtractionError,
    ResumeItem,
    ResumeOptimizerClosedError,
    ResumeOptimizerConfig,
    ResumeOptimizerError,
    ResumeOptimizerRunner,
    ResumeSection,
    TruthfulnessError,
    UnsupportedFormatError,
    create_resume_optimizer,
)
from ai_resume_optimizer.config import (
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
)


class _FakeDeepSeekModelClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.generate_calls = 0
        self.close_calls = 0

    def generate_structured(self, **kwargs: object) -> object:
        self.generate_calls += 1
        raise AssertionError("Factory tests must not execute optimization.")

    def close(self) -> None:
        self.close_calls += 1


def test_config_defaults_and_immutable_normalization() -> None:
    config = ResumeOptimizerConfig(deepseek_api_key="  unit-test-placeholder  ")

    assert config.deepseek_api_key == "unit-test-placeholder"
    assert config.deepseek_model == DEFAULT_DEEPSEEK_MODEL
    assert config.deepseek_timeout_seconds == DEFAULT_DEEPSEEK_TIMEOUT_SECONDS
    with pytest.raises(AttributeError):
        config.deepseek_model = "changed"  # type: ignore[misc]


def test_config_accepts_explicit_supported_model_and_timeout() -> None:
    config = ResumeOptimizerConfig(
        deepseek_api_key="unit-test-placeholder",
        deepseek_model=f"  {DEFAULT_DEEPSEEK_MODEL}  ",
        deepseek_timeout_seconds=12,
    )

    assert config.deepseek_model == DEFAULT_DEEPSEEK_MODEL
    assert config.deepseek_timeout_seconds == 12.0


@pytest.mark.parametrize("api_key", ["", "  "])
def test_config_rejects_blank_api_key(api_key: str) -> None:
    with pytest.raises(ConfigurationError, match="deepseek_api_key"):
        ResumeOptimizerConfig(deepseek_api_key=api_key)


@pytest.mark.parametrize("model", ["", "  ", "deepseek-chat"])
def test_config_rejects_invalid_model(model: str) -> None:
    with pytest.raises(ConfigurationError, match="deepseek_model"):
        ResumeOptimizerConfig(
            deepseek_api_key="unit-test-placeholder",
            deepseek_model=model,
        )


@pytest.mark.parametrize(
    "timeout",
    [0, -1, float("nan"), float("inf"), "not-a-number"],
)
def test_config_rejects_invalid_timeout(timeout: object) -> None:
    with pytest.raises(ConfigurationError, match="deepseek_timeout_seconds"):
        ResumeOptimizerConfig(
            deepseek_api_key="unit-test-placeholder",
            deepseek_timeout_seconds=timeout,  # type: ignore[arg-type]
        )


def test_config_repr_does_not_disclose_api_key() -> None:
    secret = "unit-test-secret-placeholder"
    config = ResumeOptimizerConfig(deepseek_api_key=secret)

    assert secret not in repr(config)
    assert config.deepseek_api_key == secret


def test_factory_composes_owned_runner_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_FakeDeepSeekModelClient] = []

    def create_client(**kwargs: object) -> _FakeDeepSeekModelClient:
        client = _FakeDeepSeekModelClient(**kwargs)  # type: ignore[arg-type]
        clients.append(client)
        return client

    def fail_getenv(name: str) -> str:
        raise AssertionError(f"Factory must not read environment variable {name}.")

    monkeypatch.setattr(factory_module, "DeepSeekModelClient", create_client)
    monkeypatch.setattr(os, "getenv", fail_getenv)
    config = ResumeOptimizerConfig(
        deepseek_api_key="unit-test-placeholder",
        deepseek_model=DEFAULT_DEEPSEEK_MODEL,
        deepseek_timeout_seconds=17,
    )

    runner = create_resume_optimizer(config)

    assert isinstance(runner, ResumeOptimizerRunner)
    assert len(clients) == 1
    assert clients[0].api_key == "unit-test-placeholder"
    assert clients[0].model == DEFAULT_DEEPSEEK_MODEL
    assert clients[0].timeout_seconds == 17.0
    assert clients[0].generate_calls == 0
    assert not list(tmp_path.iterdir())

    runner.close()
    runner.close()

    assert clients[0].close_calls == 1


def test_factory_runner_rejects_optimization_after_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_FakeDeepSeekModelClient] = []

    def create_client(**kwargs: object) -> _FakeDeepSeekModelClient:
        client = _FakeDeepSeekModelClient(**kwargs)  # type: ignore[arg-type]
        clients.append(client)
        return client

    monkeypatch.setattr(factory_module, "DeepSeekModelClient", create_client)
    runner = create_resume_optimizer(
        ResumeOptimizerConfig(deepseek_api_key="unit-test-placeholder")
    )
    runner.close()

    with pytest.raises(ResumeOptimizerClosedError):
        runner.optimize(
            resume_path=tmp_path / "resume.pdf",
            job_description="Python is required.",
        )

    assert clients[0].generate_calls == 0
    assert not list(tmp_path.iterdir())


def test_root_package_exports_stable_public_api() -> None:
    expected_exports = {
        "ConfigurationError": ConfigurationError,
        "InputError": InputError,
        "InputTooLargeError": InputTooLargeError,
        "MatchAnalysis": MatchAnalysis,
        "ModelCallError": ModelCallError,
        "ModelClient": ModelClient,
        "ModelOutputError": ModelOutputError,
        "OptimizationResult": OptimizationResult,
        "OptimizedResume": OptimizedResume,
        "OutputError": OutputError,
        "RequirementAssessment": RequirementAssessment,
        "ResumeExtractionError": ResumeExtractionError,
        "ResumeItem": ResumeItem,
        "ResumeOptimizerClosedError": ResumeOptimizerClosedError,
        "ResumeOptimizerConfig": ResumeOptimizerConfig,
        "ResumeOptimizerError": ResumeOptimizerError,
        "ResumeOptimizerRunner": ResumeOptimizerRunner,
        "ResumeSection": ResumeSection,
        "TruthfulnessError": TruthfulnessError,
        "UnsupportedFormatError": UnsupportedFormatError,
        "create_resume_optimizer": create_resume_optimizer,
    }

    for name, expected in expected_exports.items():
        assert getattr(ai_resume_optimizer, name) is expected
        assert name in ai_resume_optimizer.__all__
