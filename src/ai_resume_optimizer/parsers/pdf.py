"""Text-layer PDF resume parsing."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from ai_resume_optimizer.config import MAX_RESUME_CHARACTERS
from ai_resume_optimizer.exceptions import InputTooLargeError, ResumeExtractionError
from ai_resume_optimizer.models import ExtractedResume, SourceBlock
from ai_resume_optimizer.parsers import _normalize_extracted_text, _validate_input_file

_MIN_MEANINGFUL_CHARACTERS = 20
_MIN_PRINTABLE_RATIO = 0.8
_PDF_LAYOUT_WARNING = "Complex multi-column PDF layouts may not preserve reading order."


def _unusable_pdf_error(path: Path) -> ResumeExtractionError:
    return ResumeExtractionError(
        f"PDF '{path}' does not contain enough reliably extractable resume text. "
        "Scanned PDFs and OCR are not supported in v0.1.0; use a text-layer PDF or DOCX."
    )


def _validate_text_quality(path: Path, plain_text: str, blocks: list[SourceBlock]) -> None:
    non_whitespace = [character for character in plain_text if not character.isspace()]
    printable_count = sum(character.isprintable() for character in non_whitespace)
    printable_ratio = printable_count / len(non_whitespace) if non_whitespace else 0.0

    if (
        len(non_whitespace) < _MIN_MEANINGFUL_CHARACTERS
        or not blocks
        or not any(character.isalnum() for character in non_whitespace)
        or printable_ratio < _MIN_PRINTABLE_RATIO
    ):
        raise _unusable_pdf_error(path)


def parse_pdf_resume(path: Path) -> ExtractedResume:
    """Extract ordered paragraphs from a text-layer PDF resume."""

    _validate_input_file(path, allowed_suffixes=(".pdf",), input_name="PDF resume")

    try:
        reader = PdfReader(str(path))
        raw_pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as error:
        raise ResumeExtractionError(
            f"Could not read PDF resume '{path}'. Provide a valid, accessible text-layer PDF."
        ) from error

    if not raw_pages:
        raise _unusable_pdf_error(path)

    extracted_parts: list[tuple[int, int, int, str]] = []
    for page_number, raw_text in enumerate(raw_pages, start=1):
        page_text = _normalize_extracted_text(raw_text)
        if not page_text:
            continue
        page_parts = [part.strip() for part in page_text.split("\n\n") if part.strip()]
        for part_number, part in enumerate(page_parts, start=1):
            extracted_parts.append((page_number, part_number, len(page_parts), part))

    blocks: list[SourceBlock] = []
    for block_number, (page_number, part_number, page_part_count, text) in enumerate(
        extracted_parts,
        start=1,
    ):
        location = f"page:{page_number}"
        if page_part_count > 1:
            location = f"{location},block:{part_number}"
        blocks.append(
            SourceBlock(
                block_id=f"block-{block_number:04d}",
                text=text,
                kind="paragraph",
                location=location,
            )
        )

    plain_text = "\n\n".join(block.text for block in blocks)
    _validate_text_quality(path, plain_text, blocks)

    if len(plain_text) > MAX_RESUME_CHARACTERS:
        raise InputTooLargeError(
            f"PDF resume '{path}' contains {len(plain_text)} normalized characters; "
            f"the maximum is {MAX_RESUME_CHARACTERS}. The input will not be truncated."
        )

    return ExtractedResume(
        source_path=path,
        source_format="pdf",
        blocks=blocks,
        plain_text=plain_text,
        warnings=[_PDF_LAYOUT_WARNING],
    )


__all__ = ["parse_pdf_resume"]
