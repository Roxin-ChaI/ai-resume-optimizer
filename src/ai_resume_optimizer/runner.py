"""In-memory public runner for evidence-grounded resume optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ai_resume_optimizer.exceptions import ResumeOptimizerClosedError
from ai_resume_optimizer.model_client import ModelClient
from ai_resume_optimizer.models import OptimizationResult
from ai_resume_optimizer.pipeline import _run_optimization_in_memory


@runtime_checkable
class _ClosableModelClient(Protocol):
    def close(self) -> None:
        """Release resources owned by the model client."""


class ResumeOptimizerRunner:
    """Run resume optimization in memory through an injected model client."""

    def __init__(
        self,
        model_client: ModelClient,
        *,
        owns_model_client: bool = False,
    ) -> None:
        if owns_model_client and not isinstance(model_client, _ClosableModelClient):
            raise TypeError("An owned model_client must implement close().")

        self._model_client = model_client
        self._owned_model_client = model_client if owns_model_client else None
        self._closed = False

    def optimize(
        self,
        *,
        resume_path: Path,
        job_description: str,
    ) -> OptimizationResult:
        """Optimize one PDF or DOCX resume without writing output files."""

        if self._closed:
            raise ResumeOptimizerClosedError("Resume optimizer runner is closed.")

        return _run_optimization_in_memory(
            resume_path=resume_path,
            job_description=job_description,
            model_client=self._model_client,
        )

    def close(self) -> None:
        """Close this runner and any model client it owns."""

        if self._closed:
            return
        self._closed = True

        owned_model_client = self._owned_model_client
        if isinstance(owned_model_client, _ClosableModelClient):
            owned_model_client.close()


__all__ = ["ResumeOptimizerRunner"]
