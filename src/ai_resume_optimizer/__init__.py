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
    EvidenceSectionReference,
    MatchAnalysis,
    OptimizationResult,
    OptimizedResume,
    RequirementAssessment,
    RequirementCategory,
    RequirementEvidence,
    RequirementImportance,
    RequirementReference,
    ResumeItem,
    ResumeSection,
    SectionType,
    SourceBlockKind,
)
from ai_resume_optimizer.runner import ResumeOptimizerRunner

__version__ = "0.2.1"

__all__ = [
    "ConfigurationError",
    "EvidenceSectionReference",
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
    "RequirementCategory",
    "RequirementEvidence",
    "RequirementImportance",
    "RequirementReference",
    "ResumeExtractionError",
    "ResumeItem",
    "ResumeOptimizerConfig",
    "ResumeOptimizerClosedError",
    "ResumeOptimizerError",
    "ResumeOptimizerRunner",
    "ResumeSection",
    "SectionType",
    "SourceBlockKind",
    "TruthfulnessError",
    "UnsupportedFormatError",
    "__version__",
    "create_resume_optimizer",
]
