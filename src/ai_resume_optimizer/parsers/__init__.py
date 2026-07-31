"""Public entry points and shared helpers for supported input parsers."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from ai_resume_optimizer.exceptions import InputError, UnsupportedFormatError
from ai_resume_optimizer.models import ExtractedResume

_INLINE_WHITESPACE = re.compile(r"[^\S\n]+")


def _validate_input_file(
    path: Path,
    *,
    allowed_suffixes: tuple[str, ...],
    input_name: str,
) -> None:
    try:
        exists = path.exists()
        is_file = path.is_file()
    except OSError as error:
        raise InputError(
            f"Cannot inspect {input_name} path '{path}'. Check the path and its permissions."
        ) from error

    if not exists:
        raise InputError(f"{input_name.capitalize()} file '{path}' does not exist.")
    if not is_file:
        raise InputError(
            f"{input_name.capitalize()} path '{path}' is not a regular file. "
            "Provide a readable file."
        )
    if path.suffix.lower() not in allowed_suffixes:
        supported = ", ".join(allowed_suffixes)
        raise UnsupportedFormatError(
            f"Unsupported {input_name} format for '{path}'. Use one of: {supported}."
        )


def _normalize_extracted_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        character
        for character in normalized
        if character in {"\n", "\t"} or unicodedata.category(character) != "Cc"
    )

    lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = _INLINE_WHITESPACE.sub(" ", raw_line).rstrip()
        if line.strip():
            lines.append(line)
        elif lines and lines[-1] != "":
            lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines).strip()


def parse_resume(path: Path) -> ExtractedResume:
    """Parse a supported PDF or DOCX resume without modifying the source file."""

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from ai_resume_optimizer.parsers.pdf import parse_pdf_resume

        return parse_pdf_resume(path)
    if suffix == ".docx":
        from ai_resume_optimizer.parsers.docx import parse_docx_resume

        return parse_docx_resume(path)

    _validate_input_file(
        path,
        allowed_suffixes=(".pdf", ".docx"),
        input_name="resume",
    )
    raise AssertionError("Validated resume suffix was not dispatched.")


__all__ = ["parse_resume"]
