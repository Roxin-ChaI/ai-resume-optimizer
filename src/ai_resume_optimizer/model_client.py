"""Model-provider boundary and OpenAI structured-output adapter."""

from __future__ import annotations

import math
from typing import Protocol, TypeVar

from openai import (
    APIError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
    OpenAI,
)
from pydantic import BaseModel, ValidationError

from ai_resume_optimizer.exceptions import ModelCallError, ModelOutputError

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class ModelClient(Protocol):
    """Minimal interface required by the resume-analysis services."""

    def generate_structured(
        self,
        *,
        instructions: str,
        input_text: str,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        """Return provider output validated as the requested Pydantic model."""


class OpenAIModelClient:
    """OpenAI Responses API adapter with a provider-neutral public boundary."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        client: OpenAI | None = None,
    ) -> None:
        normalized_api_key = api_key.strip()
        normalized_model = model.strip()
        timeout = float(timeout_seconds)

        if not normalized_api_key:
            raise ValueError("api_key must not be blank.")
        if not normalized_model:
            raise ValueError("model must not be blank.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds must be a positive finite number.")

        self._model = normalized_model
        self._client = (
            client
            if client is not None
            else OpenAI(
                api_key=normalized_api_key,
                timeout=timeout,
                max_retries=0,
            )
        )

    def generate_structured(
        self,
        *,
        instructions: str,
        input_text: str,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        """Generate and locally revalidate one structured response."""

        normalized_instructions = instructions.strip()
        normalized_input = input_text.strip()
        if not normalized_instructions:
            raise ValueError("instructions must not be blank.")
        if not normalized_input:
            raise ValueError("input_text must not be blank.")

        try:
            response = self._client.responses.parse(
                model=self._model,
                instructions=normalized_instructions,
                input=normalized_input,
                text_format=response_model,
                store=False,
            )
        except (LengthFinishReasonError, ContentFilterFinishReasonError) as error:
            raise ModelOutputError(
                "The model response could not be parsed as the required structure."
            ) from error
        except APIError as error:
            raise ModelCallError("The model provider request failed.") from error

        try:
            parsed = response.output_parsed
            if parsed is None:
                raise ModelOutputError(
                    "The model response did not contain the required structured output."
                )
            if not isinstance(parsed, response_model):
                raise ModelOutputError(
                    "The model response used an unexpected structured output type."
                )
            return response_model.model_validate(parsed.model_dump())
        except ModelOutputError:
            raise
        except (AttributeError, ValidationError) as error:
            raise ModelOutputError(
                "The model response failed local structured-output validation."
            ) from error
