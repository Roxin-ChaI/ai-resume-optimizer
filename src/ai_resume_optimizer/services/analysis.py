"""Structured resume and job-description analysis services."""

from __future__ import annotations

import json
import re

from ai_resume_optimizer.exceptions import ModelOutputError
from ai_resume_optimizer.model_client import ModelClient
from ai_resume_optimizer.models import (
    ExtractedResume,
    JobProfile,
    MatchAnalysis,
    OptimizedResume,
    ResumeItem,
    StructuredResume,
)
from ai_resume_optimizer.prompts import load_prompt

_NUMERIC_SCORE_PATTERN = re.compile(r"(?<!\d)\d+(?:\.\d+)?\s*%")
_ATS_SCORE_PATTERN = re.compile(r"ATS\s*score|ATS\s*分数", re.IGNORECASE)
_PREDICTION_PATTERN = re.compile(r"通过率|录取率|招聘平台评分")


def _iter_items(structured_resume: StructuredResume) -> list[ResumeItem]:
    return [
        *(item for section in structured_resume.sections for item in section.items),
        *structured_resume.unclassified_content,
    ]


def _validate_known_source_block_ids(
    extracted_resume: ExtractedResume,
    structured_resume: StructuredResume,
) -> None:
    valid_ids = {block.block_id for block in extracted_resume.blocks}

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


def _restore_missing_source_blocks(
    extracted_resume: ExtractedResume,
    structured_resume: StructuredResume,
) -> StructuredResume:
    valid_ids = {block.block_id for block in extracted_resume.blocks}
    referenced_ids = {
        block_id for item in _iter_items(structured_resume) for block_id in item.source_block_ids
    }
    missing_ids = valid_ids - referenced_ids
    restored_items = [
        ResumeItem(
            text=block.text,
            source_block_ids=[block.block_id],
            related_requirement_ids=[],
            needs_review=False,
            review_note=None,
        )
        for block in extracted_resume.blocks
        if block.block_id in missing_ids
    ]
    if not restored_items:
        return structured_resume

    return structured_resume.model_copy(
        update={
            "unclassified_content": [
                *structured_resume.unclassified_content,
                *restored_items,
            ]
        }
    )


def _validate_structured_resume(
    extracted_resume: ExtractedResume,
    structured_resume: StructuredResume,
) -> None:
    valid_ids = {block.block_id for block in extracted_resume.blocks}
    item_ids: set[str] = set()

    _validate_known_source_block_ids(extracted_resume, structured_resume)
    for item in _iter_items(structured_resume):
        for block_id in item.source_block_ids:
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


def _build_structure_resume_input(extracted_resume: ExtractedResume) -> str:
    required_ids = [block.block_id for block in extracted_resume.blocks]
    source_data = {
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
    return "\n".join(
        [
            f"REQUIRED_SOURCE_BLOCK_COUNT: {len(required_ids)}",
            "REQUIRED_SOURCE_BLOCK_IDS:",
            json.dumps(required_ids, ensure_ascii=False),
            "SOURCE_BLOCKS:",
            json.dumps(source_data, ensure_ascii=False),
        ]
    )


def structure_resume(
    extracted_resume: ExtractedResume,
    model_client: ModelClient,
) -> StructuredResume:
    """Structure ordered resume blocks while retaining complete source coverage."""

    result = model_client.generate_structured(
        instructions=load_prompt("structure_resume.txt"),
        input_text=_build_structure_resume_input(extracted_resume),
        response_model=StructuredResume,
    )
    _validate_known_source_block_ids(extracted_resume, result)
    complete_result = _restore_missing_source_blocks(extracted_resume, result)
    _validate_structured_resume(extracted_resume, complete_result)
    return complete_result


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


def _structured_source_block_ids(structured_resume: StructuredResume) -> set[str]:
    return {
        block_id for item in _iter_items(structured_resume) for block_id in item.source_block_ids
    }


def _validate_assessment_coverage(
    job_profile: JobProfile,
    match_analysis: MatchAnalysis,
) -> None:
    expected_ids = [requirement.requirement_id for requirement in job_profile.requirements]
    actual_ids = [assessment.requirement_id for assessment in match_analysis.assessments]
    if actual_ids != expected_ids:
        raise ModelOutputError(
            "Match assessments must cover every job requirement exactly once and in order."
        )


def _validate_match_analysis(
    structured_resume: StructuredResume,
    job_profile: JobProfile,
    match_analysis: MatchAnalysis,
) -> None:
    _validate_assessment_coverage(job_profile, match_analysis)
    valid_block_ids = _structured_source_block_ids(structured_resume)
    for assessment in match_analysis.assessments:
        for block_id in assessment.source_block_ids:
            if block_id not in valid_block_ids:
                raise ModelOutputError(
                    f"Match assessment {assessment.requirement_id!r} references "
                    f"unknown source block ID {block_id!r}."
                )

    evaluation = match_analysis.overall_evaluation
    if (
        _NUMERIC_SCORE_PATTERN.search(evaluation)
        or _ATS_SCORE_PATTERN.search(evaluation)
        or _PREDICTION_PATTERN.search(evaluation)
    ):
        raise ModelOutputError(
            "The overall match evaluation must not contain a numeric score, "
            "ATS score, or recruiting prediction."
        )


def _build_match_analysis_input(
    structured_resume: StructuredResume,
    job_profile: JobProfile,
) -> str:
    required_ids = [requirement.requirement_id for requirement in job_profile.requirements]
    match_input = {
        "structured_resume": structured_resume.model_dump(mode="json"),
        "job_profile": job_profile.model_dump(mode="json"),
    }
    return "\n".join(
        [
            f"REQUIRED_REQUIREMENT_COUNT: {len(required_ids)}",
            "REQUIRED_REQUIREMENT_IDS:",
            json.dumps(required_ids, ensure_ascii=False),
            "MATCH_INPUT:",
            json.dumps(match_input, ensure_ascii=False),
        ]
    )


def analyze_match(
    model_client: ModelClient,
    structured_resume: StructuredResume,
    job_profile: JobProfile,
) -> MatchAnalysis:
    """Analyze resume evidence against every job requirement."""

    result = model_client.generate_structured(
        instructions=load_prompt("analyze_match.txt"),
        input_text=_build_match_analysis_input(structured_resume, job_profile),
        response_model=MatchAnalysis,
    )
    _validate_match_analysis(structured_resume, job_profile, result)
    return result


def _validate_optimized_resume_relationships(
    structured_resume: StructuredResume,
    job_profile: JobProfile,
    match_analysis: MatchAnalysis,
    optimized_resume: OptimizedResume,
) -> None:
    valid_block_ids = _structured_source_block_ids(structured_resume)
    valid_requirement_ids = {requirement.requirement_id for requirement in job_profile.requirements}
    unsupported_ids = {
        assessment.requirement_id
        for assessment in match_analysis.assessments
        if assessment.status == "unsupported"
    }

    for section in optimized_resume.sections:
        for block_id in section.source_block_ids:
            if block_id not in valid_block_ids:
                raise ModelOutputError(
                    f"Optimized resume references unknown source block ID {block_id!r}."
                )
        for item in section.items:
            for block_id in item.source_block_ids:
                if block_id not in valid_block_ids:
                    raise ModelOutputError(
                        f"Optimized resume references unknown source block ID {block_id!r}."
                    )
            for requirement_id in item.related_requirement_ids:
                if requirement_id not in valid_requirement_ids:
                    raise ModelOutputError(
                        f"Optimized resume references unknown requirement ID {requirement_id!r}."
                    )
                if requirement_id in unsupported_ids:
                    raise ModelOutputError(
                        f"Optimized resume references unsupported requirement {requirement_id!r}."
                    )

    if any(not item.strip() for item in optimized_resume.pending_user_inputs):
        raise ModelOutputError("Pending user inputs must not contain blank entries.")


def optimize_resume(
    model_client: ModelClient,
    structured_resume: StructuredResume,
    job_profile: JobProfile,
    match_analysis: MatchAnalysis,
) -> OptimizedResume:
    """Produce an evidence-linked optimized resume representation."""

    _validate_assessment_coverage(job_profile, match_analysis)
    input_data = {
        "structured_resume": structured_resume.model_dump(mode="json"),
        "job_profile": job_profile.model_dump(mode="json"),
        "match_analysis": match_analysis.model_dump(mode="json"),
    }
    result = model_client.generate_structured(
        instructions=load_prompt("optimize_resume.txt"),
        input_text=json.dumps(input_data, ensure_ascii=False),
        response_model=OptimizedResume,
    )
    _validate_optimized_resume_relationships(
        structured_resume,
        job_profile,
        match_analysis,
        result,
    )
    return result
