"""Deterministic model-client fake with call recording."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar, cast

from pydantic import BaseModel

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


@dataclass(frozen=True)
class ModelCallRecord:
    """One structured-generation call."""

    instructions: str
    input_text: str
    response_model: type[BaseModel]


class FakeModelClient:
    """Return configured Pydantic values without SDK or network access."""

    def __init__(
        self,
        responses: dict[type[BaseModel], BaseModel] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.responses = responses or {}
        self.error = error
        self.calls: list[ModelCallRecord] = []

    def generate_structured(
        self,
        *,
        instructions: str,
        input_text: str,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        self.calls.append(
            ModelCallRecord(
                instructions=instructions,
                input_text=input_text,
                response_model=response_model,
            )
        )
        if self.error is not None:
            raise self.error
        if response_model not in self.responses:
            raise AssertionError(f"No fake response configured for {response_model.__name__}.")

        response = self.responses[response_model]
        if not isinstance(response, response_model):
            raise TypeError(f"Configured response is not an instance of {response_model.__name__}.")
        return cast(ResponseModelT, response)
