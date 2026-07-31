"""Environment-backed application settings."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from ai_resume_optimizer.exceptions import ConfigurationError

MAX_RESUME_CHARACTERS = 50_000
MAX_JOB_DESCRIPTION_CHARACTERS = 30_000
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_DEEPSEEK_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings required to construct external service clients."""

    deepseek_api_key: str
    deepseek_model: str
    deepseek_timeout_seconds: float


def _required_environment_value(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ConfigurationError(f"{name} is required and must not be blank.")
    return value.strip()


def _deepseek_model() -> str:
    raw_value = os.getenv("DEEPSEEK_MODEL")
    if raw_value is None:
        return DEFAULT_DEEPSEEK_MODEL

    model = raw_value.strip()
    if model != DEFAULT_DEEPSEEK_MODEL:
        raise ConfigurationError(
            f"DEEPSEEK_MODEL must be {DEFAULT_DEEPSEEK_MODEL!r}; other models are unsupported."
        )
    return model


def _deepseek_timeout_seconds() -> float:
    raw_value = os.getenv("DEEPSEEK_TIMEOUT_SECONDS")
    if raw_value is None:
        return DEFAULT_DEEPSEEK_TIMEOUT_SECONDS

    try:
        timeout = float(raw_value.strip())
    except ValueError as error:
        raise ConfigurationError("DEEPSEEK_TIMEOUT_SECONDS must be a number.") from error

    if not math.isfinite(timeout) or timeout <= 0:
        raise ConfigurationError("DEEPSEEK_TIMEOUT_SECONDS must be a positive finite number.")
    return timeout


def load_settings() -> Settings:
    """Read and validate settings without retaining global client state."""

    return Settings(
        deepseek_api_key=_required_environment_value("DEEPSEEK_API_KEY"),
        deepseek_model=_deepseek_model(),
        deepseek_timeout_seconds=_deepseek_timeout_seconds(),
    )
