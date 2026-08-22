"""Project-specific exceptions with stable CLI-oriented exit categories."""


class ResumeOptimizerError(Exception):
    """Base class for expected application errors."""

    exit_code = 1


class ResumeOptimizerClosedError(ResumeOptimizerError):
    """Raised when an optimization is requested after its runner is closed."""


class ConfigurationError(ResumeOptimizerError):
    """Raised when required application configuration is invalid."""

    exit_code = 4


class InputError(ResumeOptimizerError):
    """Raised when a user-provided input is missing or invalid."""

    exit_code = 2


class UnsupportedFormatError(InputError):
    """Raised when an input format is outside the supported v0.2.0 scope."""


class InputTooLargeError(InputError):
    """Raised when normalized input exceeds an approved character limit."""


class ResumeExtractionError(ResumeOptimizerError):
    """Raised when usable resume text cannot be extracted."""

    exit_code = 3


class ModelCallError(ResumeOptimizerError):
    """Raised when a configured model provider call fails."""

    exit_code = 4


class ModelOutputError(ResumeOptimizerError):
    """Raised when model output cannot be validated."""

    exit_code = 5


class TruthfulnessError(ResumeOptimizerError):
    """Raised when optimized content violates an evidence constraint."""

    exit_code = 5


class OutputError(ResumeOptimizerError):
    """Raised when a requested output cannot be created safely."""

    exit_code = 6
