"""Structured resume and job-description analysis services."""

from __future__ import annotations

import json

from ai_resume_optimizer.exceptions import ModelOutputError
from ai_resume_optimizer.model_client import ModelClient
from ai_resume_optimizer.models import (
    ExtractedResume,
    JobProfile,
    ResumeItem,
    StructuredResume,
)
from ai_resume_optimizer.prompts import load_prompt


def _iter_items(structured_resume: StructuredResume) -> list[ResumeItem]:
    return [
        *(item for section in structured_resume.sections for item in section.items),
        *structured_resume.unclassified_content,
    ]


def _validate_structured_resume(
    extracted_resume: ExtractedResume,
    structured_resume: StructuredResume,
) -> None:
    valid_ids = {block.block_id for block in extracted_resume.blocks}
    item_ids: set[str] = set()

    for section in structured_resume.sections:
        for block_id in section.source_block_ids:
            if block_id not in valid_ids:
                raise ModelOutputError(
                    f"Structured resume references unknown source block ID {block_id!r}."
                )

    for item in _iter_items(structured_resume):
        for block_id in item.source_block_ids:
            if block_id not in valid_ids:
                raise ModelOutputError(
                    f"Structured resume references unknown source block ID {block_id!r}."
                )
            item_ids.add(block_id)
        if item.related_requirement_ids:
            raise ModelOutputError("Resume structuring must not add job requirement relationships.")
        if item.needs_review or item.review_note is not None:
            raise ModelOutputError("Resume structuring must not add review-state fields.")

    missing_ids = valid_ids - item_ids
    if missing_ids:
        missing_id = next(
            block.block_id for block in extracted_resume.blocks if block.block_id in missing_ids
        )
        raise ModelOutputError(f"Structured resume omitted source block ID {missing_id!r}.")


def structure_resume(
    extracted_resume: ExtractedResume,
    model_client: ModelClient,
) -> StructuredResume:
    """Structure ordered resume blocks while retaining complete source coverage."""

    input_data = {
        "source_format": extracted_resume.source_format,
        "blocks": [
            {
                "block_id": block.block_id,
                "kind": block.kind,
                "location": block.location,
                "text": block.text,
            }
            for block in extracted_resume.blocks
        ],
    }
    result = model_client.generate_structured(
        instructions=load_prompt("structure_resume.txt"),
        input_text=json.dumps(input_data, ensure_ascii=False),
        response_model=StructuredResume,
    )
    _validate_structured_resume(extracted_resume, result)
    return result


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.replace("\r\n", "\n").replace("\r", "\n").split())


def _validate_job_profile(job_description: str, job_profile: JobProfile) -> None:
    normalized_description = _normalize_whitespace(job_description)
    for index, requirement in enumerate(job_profile.requirements, start=1):
        expected_id = f"requirement-{index:04d}"
        if requirement.requirement_id != expected_id:
            raise ModelOutputError(
                f"Job requirement {requirement.requirement_id!r} has an invalid sequence ID."
            )
        normalized_excerpt = _normalize_whitespace(requirement.source_excerpt)
        if normalized_excerpt not in normalized_description:
            raise ModelOutputError(
                f"Job requirement {requirement.requirement_id!r} has no exact source excerpt."
            )


def analyze_job(job_description: str, model_client: ModelClient) -> JobProfile:
    """Extract explicit requirements from a normalized job description."""

    if not isinstance(job_description, str):
        raise ValueError("job_description must be a string.")
    normalized_description = job_description.strip()
    if not normalized_description:
        raise ValueError("job_description must not be blank.")

    result = model_client.generate_structured(
        instructions=load_prompt("analyze_job.txt"),
        input_text=normalized_description,
        response_model=JobProfile,
    )
    _validate_job_profile(normalized_description, result)
    return result
