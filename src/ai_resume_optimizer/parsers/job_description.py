"""TXT job-description reading and deterministic text normalization."""

from __future__ import annotations

from pathlib import Path

from ai_resume_optimizer.config import MAX_JOB_DESCRIPTION_CHARACTERS
from ai_resume_optimizer.exceptions import InputError, InputTooLargeError
from ai_resume_optimizer.parsers import _normalize_extracted_text, _validate_input_file


def normalize_job_description(text: str) -> str:
    """Normalize directly supplied job-description text without changing its meaning."""

    normalized = _normalize_extracted_text(text)
    if not normalized:
        raise InputError("Job description is empty. Provide non-empty job-description text.")
    if len(normalized) > MAX_JOB_DESCRIPTION_CHARACTERS:
        raise InputTooLargeError(
            f"Job description contains {len(normalized)} normalized characters; "
            f"the maximum is {MAX_JOB_DESCRIPTION_CHARACTERS}. "
            "The input will not be truncated."
        )
    return normalized


def read_job_description(path: Path) -> str:
    """Read and normalize a UTF-8 or UTF-8 BOM job-description text file."""

    _validate_input_file(path, allowed_suffixes=(".txt",), input_name="job description")

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise InputError(
            f"Job description file '{path}' is not valid UTF-8. Save it as UTF-8 text."
        ) from error
    except OSError as error:
        raise InputError(
            f"Could not read job description file '{path}'. Check its permissions."
        ) from error

    return normalize_job_description(text)


__all__ = ["normalize_job_description", "read_job_description"]
