"""Model-provider boundary and DeepSeek JSON-output adapter."""

from __future__ import annotations

import json
import math
from typing import Protocol, TypeVar

from openai import APIError, OpenAI
from pydantic import BaseModel, ValidationError

from ai_resume_optimizer.config import DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_MODEL
from ai_resume_optimizer.exceptions import ModelCallError, ModelOutputError

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)
_MAX_VALIDATION_ISSUES = 5


def _format_validation_location(location: tuple[object, ...]) -> str:
    parts = [str(part) if isinstance(part, (str, int)) else "<field>" for part in location]
    return ".".join(parts) if parts else "<root>"


def _format_validation_error(
    response_model: type[BaseModel],
    error: ValidationError,
) -> str:
    issues = error.errors(include_input=False, include_url=False)
    formatted_issues: list[str] = []
    for issue in issues[:_MAX_VALIDATION_ISSUES]:
        location = issue.get("loc", ())
        safe_location = (
            _format_validation_location(location) if isinstance(location, tuple) else "<root>"
        )
        error_type = issue.get("type", "validation_error")
        if not isinstance(error_type, str):
            error_type = "validation_error"
        message = issue.get("msg", "Validation failed.")
        if not isinstance(message, str):
            message = "Validation failed."
        formatted_issues.append(f"{safe_location} [{error_type}]: {message}")

    remaining_count = len(issues) - len(formatted_issues)
    if remaining_count:
        formatted_issues.append(f"(+{remaining_count} more validation errors)")
    details = "; ".join(formatted_issues) or "unknown validation error"
    return f"{response_model.__name__} output validation failed: {details}"


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


class DeepSeekModelClient:
    """DeepSeek Chat Completions adapter with a provider-neutral public boundary."""

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
        if normalized_model != DEFAULT_DEEPSEEK_MODEL:
            raise ValueError(f"model must be {DEFAULT_DEEPSEEK_MODEL!r}.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds must be a positive finite number.")

        self._model = normalized_model
        self._closed = False
        self._client = (
            client
            if client is not None
            else OpenAI(
                api_key=normalized_api_key,
                base_url=DEEPSEEK_BASE_URL,
                timeout=timeout,
                max_retries=0,
            )
        )

    @staticmethod
    def _system_content(instructions: str, response_model: type[BaseModel]) -> str:
        schema_json = json.dumps(
            response_model.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
        )
        return (
            f"{instructions}\n\n"
            "JSON Output rules:\n"
            "- Output exactly one JSON object.\n"
            "- Do not output Markdown or code fences.\n"
            "- Do not output a preamble, explanation, or postscript.\n"
            "- The JSON object must conform exactly to the provided JSON Schema.\n\n"
            f"JSON Schema:\n{schema_json}"
        )

    def generate_structured(
        self,
        *,
        instructions: str,
        input_text: str,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        """Generate and locally revalidate one structured response."""

        response_model_name = response_model.__name__
        normalized_instructions = instructions.strip()
        if not normalized_instructions:
            raise ValueError("instructions must not be blank.")
        if not input_text.strip():
            raise ValueError("input_text must not be blank.")

        system_content = self._system_content(normalized_instructions, response_model)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": input_text},
                ],
                response_format={"type": "json_object"},
                stream=False,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except APIError as error:
            raise ModelCallError("The model provider request failed.") from error

        try:
            choices = response.choices
            if not choices:
                raise ModelOutputError(
                    f"{response_model_name} output validation failed: "
                    "the model returned no choices."
                )

            choice = choices[0]
            if choice.finish_reason == "length":
                raise ModelOutputError(
                    f"{response_model_name} output validation failed: the response was truncated."
                )

            message = choice.message
            if message is None:
                raise ModelOutputError(
                    f"{response_model_name} output validation failed: "
                    "the model returned no message."
                )
            content = message.content
            if content is None or not isinstance(content, str) or not content.strip():
                raise ModelOutputError(
                    f"{response_model_name} output validation failed: "
                    "the model returned empty content."
                )

            parsed_json = json.loads(content)
            if not isinstance(parsed_json, dict):
                received_type = type(parsed_json).__name__
                raise ModelOutputError(
                    f"{response_model_name} output validation failed: "
                    f"expected a JSON object, received {received_type}."
                )
            return response_model.model_validate(parsed_json)
        except ModelOutputError:
            raise
        except json.JSONDecodeError as error:
            raise ModelOutputError(
                f"{response_model_name} JSON decoding failed at "
                f"line {error.lineno}, column {error.colno}."
            ) from error
        except ValidationError as error:
            raise ModelOutputError(_format_validation_error(response_model, error)) from error
        except (AttributeError, IndexError, TypeError) as error:
            raise ModelOutputError(
                f"{response_model_name} output validation failed: "
                "the response structure could not be inspected."
            ) from error

    def close(self) -> None:
        """Close the underlying SDK client at most once."""

        if self._closed:
            return
        self._closed = True
        self._client.close()
