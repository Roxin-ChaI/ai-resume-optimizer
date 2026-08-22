"""AI Resume Optimizer package."""

from ai_resume_optimizer.exceptions import (
    ConfigurationError,
    InputError,
    InputTooLargeError,
    ModelCallError,
    ModelOutputError,
    OutputError,
    ResumeExtractionError,
    ResumeOptimizerClosedError,
    ResumeOptimizerError,
    TruthfulnessError,
    UnsupportedFormatError,
)
from ai_resume_optimizer.factory import (
    ResumeOptimizerConfig,
    create_resume_optimizer,
)
from ai_resume_optimizer.model_client import ModelClient
from ai_resume_optimizer.models import (
    MatchAnalysis,
    OptimizationResult,
    OptimizedResume,
    RequirementAssessment,
    ResumeItem,
    ResumeSection,
)
from ai_resume_optimizer.runner import ResumeOptimizerRunner

__version__ = "0.2.0"

__all__ = [
    "ConfigurationError",
    "InputError",
    "InputTooLargeError",
    "MatchAnalysis",
    "ModelCallError",
    "ModelClient",
    "ModelOutputError",
    "OptimizationResult",
    "OptimizedResume",
    "OutputError",
    "RequirementAssessment",
    "ResumeExtractionError",
    "ResumeItem",
    "ResumeOptimizerConfig",
    "ResumeOptimizerClosedError",
    "ResumeOptimizerError",
    "ResumeOptimizerRunner",
    "ResumeSection",
    "TruthfulnessError",
    "UnsupportedFormatError",
    "__version__",
    "create_resume_optimizer",
]
