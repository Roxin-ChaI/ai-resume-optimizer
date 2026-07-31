"""DOCX resume parsing with body-order preservation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from docx import Document
from docx.document import Document as DocumentObject
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from ai_resume_optimizer.config import MAX_RESUME_CHARACTERS
from ai_resume_optimizer.exceptions import InputTooLargeError, ResumeExtractionError
from ai_resume_optimizer.models import ExtractedResume, SourceBlock
from ai_resume_optimizer.parsers import _normalize_extracted_text, _validate_input_file

ParagraphKind = Literal["paragraph", "heading", "list_item"]


def _paragraph_kind(paragraph: Paragraph) -> ParagraphKind:
    style_name = paragraph.style.name.casefold() if paragraph.style else ""
    if style_name.startswith("heading"):
        return "heading"

    paragraph_properties = paragraph._p.pPr
    has_numbering = paragraph_properties is not None and paragraph_properties.numPr is not None
    if style_name.startswith("list") or has_numbering:
        return "list_item"
    return "paragraph"


def _table_row_texts(table: Table) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    seen_cell_elements: list[object] = []

    for row_number, row in enumerate(table.rows, start=1):
        cell_texts: list[str] = []
        for cell in row.cells:
            cell_element = cell._tc
            if any(cell_element is seen for seen in seen_cell_elements):
                continue
            seen_cell_elements.append(cell_element)

            paragraphs = [
                normalized
                for paragraph in cell.paragraphs
                if (normalized := _normalize_extracted_text(paragraph.text))
            ]
            if paragraphs:
                cell_texts.append(" / ".join(paragraphs))

        if cell_texts:
            rows.append((row_number, " | ".join(cell_texts)))
    return rows


def _extract_body_entries(
    document: DocumentObject,
) -> list[tuple[str, Literal["paragraph", "heading", "list_item", "table_row"], str]]:
    entries: list[tuple[str, Literal["paragraph", "heading", "list_item", "table_row"], str]] = []
    body_block_number = 0
    table_number = 0

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            body_block_number += 1
            paragraph = Paragraph(child, document)
            text = _normalize_extracted_text(paragraph.text)
            if text:
                entries.append(
                    (text, _paragraph_kind(paragraph), f"body:block:{body_block_number}")
                )
        elif isinstance(child, CT_Tbl):
            body_block_number += 1
            table_number += 1
            table = Table(child, document)
            for row_number, text in _table_row_texts(table):
                entries.append((text, "table_row", f"table:{table_number},row:{row_number}"))

    return entries


def parse_docx_resume(path: Path) -> ExtractedResume:
    """Extract paragraphs and table rows from a DOCX in body order."""

    _validate_input_file(path, allowed_suffixes=(".docx",), input_name="DOCX resume")

    try:
        document = Document(str(path))
        entries = _extract_body_entries(document)
    except Exception as error:
        raise ResumeExtractionError(
            f"Could not read DOCX resume '{path}'. Provide a valid, accessible DOCX file."
        ) from error

    if not entries:
        raise ResumeExtractionError(
            f"DOCX resume '{path}' does not contain any valid resume text. "
            "Provide a DOCX with readable paragraphs or table content."
        )

    blocks = [
        SourceBlock(
            block_id=f"block-{block_number:04d}",
            text=text,
            kind=kind,
            location=location,
        )
        for block_number, (text, kind, location) in enumerate(entries, start=1)
    ]
    plain_text = "\n\n".join(block.text for block in blocks)

    if not plain_text:
        raise ResumeExtractionError(f"DOCX resume '{path}' does not contain any valid resume text.")
    if len(plain_text) > MAX_RESUME_CHARACTERS:
        raise InputTooLargeError(
            f"DOCX resume '{path}' contains {len(plain_text)} normalized characters; "
            f"the maximum is {MAX_RESUME_CHARACTERS}. The input will not be truncated."
        )

    return ExtractedResume(
        source_path=path,
        source_format="docx",
        blocks=blocks,
        plain_text=plain_text,
        warnings=[],
    )


__all__ = ["parse_docx_resume"]
