"""Production configuration and composition for the public optimizer runner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ai_resume_optimizer.config import (
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_DEEPSEEK_TIMEOUT_SECONDS,
)
from ai_resume_optimizer.exceptions import ConfigurationError
from ai_resume_optimizer.model_client import DeepSeekModelClient
from ai_resume_optimizer.runner import ResumeOptimizerRunner


@dataclass(frozen=True, slots=True)
class ResumeOptimizerConfig:
    """Validated configuration for the production DeepSeek composition."""

    deepseek_api_key: str = field(repr=False)
    deepseek_model: str = DEFAULT_DEEPSEEK_MODEL
    deepseek_timeout_seconds: float = DEFAULT_DEEPSEEK_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        api_key = self.deepseek_api_key
        if not isinstance(api_key, str) or not api_key.strip():
            raise ConfigurationError("deepseek_api_key is required and must not be blank.")

        model = self.deepseek_model
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError("deepseek_model is required and must not be blank.")
        normalized_model = model.strip()
        if normalized_model != DEFAULT_DEEPSEEK_MODEL:
            raise ConfigurationError(
                f"deepseek_model must be {DEFAULT_DEEPSEEK_MODEL!r}; other models are unsupported."
            )

        try:
            timeout = float(self.deepseek_timeout_seconds)
        except (TypeError, ValueError) as error:
            raise ConfigurationError("deepseek_timeout_seconds must be a number.") from error
        if not math.isfinite(timeout) or timeout <= 0:
            raise ConfigurationError("deepseek_timeout_seconds must be a positive finite number.")

        object.__setattr__(self, "deepseek_api_key", api_key.strip())
        object.__setattr__(self, "deepseek_model", normalized_model)
        object.__setattr__(self, "deepseek_timeout_seconds", timeout)


def create_resume_optimizer(config: ResumeOptimizerConfig) -> ResumeOptimizerRunner:
    """Create a production optimizer runner backed by DeepSeek."""

    model_client = DeepSeekModelClient(
        api_key=config.deepseek_api_key,
        model=config.deepseek_model,
        timeout_seconds=config.deepseek_timeout_seconds,
    )
    return ResumeOptimizerRunner(model_client, owns_model_client=True)


__all__ = ["ResumeOptimizerConfig", "create_resume_optimizer"]
