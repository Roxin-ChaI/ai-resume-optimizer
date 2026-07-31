"""Environment-backed application settings."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from ai_resume_optimizer.exceptions import ConfigurationError

MAX_RESUME_CHARACTERS = 50_000
MAX_JOB_DESCRIPTION_CHARACTERS = 30_000
DEFAULT_OPENAI_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated settings required to construct external service clients."""

    openai_api_key: str
    openai_model: str
    openai_timeout_seconds: float


def _required_environment_value(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ConfigurationError(f"{name} is required and must not be blank.")
    return value.strip()


def _openai_timeout_seconds() -> float:
    raw_value = os.getenv("OPENAI_TIMEOUT_SECONDS")
    if raw_value is None:
        return DEFAULT_OPENAI_TIMEOUT_SECONDS

    try:
        timeout = float(raw_value.strip())
    except ValueError as error:
        raise ConfigurationError("OPENAI_TIMEOUT_SECONDS must be a number.") from error

    if not math.isfinite(timeout) or timeout <= 0:
        raise ConfigurationError("OPENAI_TIMEOUT_SECONDS must be a positive finite number.")
    return timeout


def load_settings() -> Settings:
    """Read and validate settings without retaining global client state."""

    return Settings(
        openai_api_key=_required_environment_value("OPENAI_API_KEY"),
        openai_model=_required_environment_value("OPENAI_MODEL"),
        openai_timeout_seconds=_openai_timeout_seconds(),
    )
