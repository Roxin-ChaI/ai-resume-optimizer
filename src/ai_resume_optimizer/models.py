"""Strict data models shared by the AI Resume Optimizer workflow."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NonEmptyString = Annotated[str, Field(min_length=1)]
SourceBlockKind = Literal["paragraph", "heading", "list_item", "table_row"]
ResumeFormat = Literal["pdf", "docx"]
SectionType = Literal[
    "basic_info",
    "summary",
    "skills",
    "experience",
    "projects",
    "education",
    "certifications",
    "other",
]
RequirementCategory = Literal[
    "core_skill",
    "preferred_skill",
    "experience",
    "responsibility",
    "education_or_qualification",
    "keyword",
]
RequirementImportance = Literal["required", "preferred", "contextual"]
AssessmentStatus = Literal["well_supported", "underrepresented", "unsupported"]
OverallRating = Literal["高", "一般", "低"]
OutputPathKey = Literal[
    "analysis_report",
    "optimized_resume_markdown",
    "optimized_resume_docx",
]


def _clean_string_list(value: object) -> object:
    if not isinstance(value, list):
        return value
    return [
        item.strip() if isinstance(item, str) else item
        for item in value
        if not isinstance(item, str) or item.strip()
    ]


def _deduplicate_ids(value: object) -> object:
    if not isinstance(value, list):
        return value

    result: list[object] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            result.append(item)
            continue

        normalized = item.strip()
        if not normalized:
            raise ValueError("IDs must not be blank.")
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _ordered_source_block_ids(items: list[ResumeItem]) -> list[str]:
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for item in items:
        for block_id in item.source_block_ids:
            if block_id not in seen:
                seen.add(block_id)
                ordered_ids.append(block_id)
    return ordered_ids


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and normalizes string whitespace."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SourceBlock(StrictModel):
    """An ordered unit of text extracted from a source resume."""

    block_id: NonEmptyString
    text: NonEmptyString
    kind: SourceBlockKind
    location: NonEmptyString


class ExtractedResume(StrictModel):
    """Text blocks extracted from a supported resume file."""

    source_path: Path
    source_format: ResumeFormat
    blocks: Annotated[list[SourceBlock], Field(min_length=1)]
    plain_text: NonEmptyString
    warnings: list[str]

    @field_validator("warnings", mode="before")
    @classmethod
    def clean_warnings(cls, value: object) -> object:
        """Remove blank warning entries while preserving valid messages."""

        return _clean_string_list(value)

    @model_validator(mode="after")
    def validate_unique_block_ids(self) -> ExtractedResume:
        block_ids = [block.block_id for block in self.blocks]
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("Source block IDs must be unique within a resume.")
        return self


class ResumeItem(StrictModel):
    """A factual resume item linked to one or more source blocks."""

    text: NonEmptyString
    source_block_ids: Annotated[list[str], Field(min_length=1)]
    related_requirement_ids: list[str]
    needs_review: bool
    review_note: NonEmptyString | None

    @field_validator("source_block_ids", "related_requirement_ids", mode="before")
    @classmethod
    def normalize_ids(cls, value: object) -> object:
        """Reject blank IDs and deduplicate valid IDs in first-seen order."""

        return _deduplicate_ids(value)

    @model_validator(mode="after")
    def validate_review_state(self) -> ResumeItem:
        if self.needs_review and self.review_note is None:
            raise ValueError("review_note is required when needs_review is true.")
        if not self.needs_review and self.review_note is not None:
            raise ValueError("review_note must be null when needs_review is false.")
        return self


class ResumeSection(StrictModel):
    """A non-empty resume section with an exact aggregate evidence set."""

    section_type: SectionType
    title: NonEmptyString
    items: Annotated[list[ResumeItem], Field(min_length=1)]
    source_block_ids: list[str]

    @model_validator(mode="before")
    @classmethod
    def derive_source_id_union(cls, value: object) -> object:
        """Derive section evidence IDs from item evidence without mutating input."""

        if not isinstance(value, Mapping):
            return value
        items = value.get("items")
        if not isinstance(items, list):
            return value

        ordered_ids: list[str] = []
        seen: set[str] = set()
        for item in items:
            if isinstance(item, ResumeItem):
                source_ids: object = item.source_block_ids
            elif isinstance(item, Mapping):
                source_ids = item.get("source_block_ids")
            else:
                return value
            if not isinstance(source_ids, list):
                return value

            for block_id in source_ids:
                if not isinstance(block_id, str) or not block_id.strip():
                    return value
                normalized_id = block_id.strip()
                if normalized_id not in seen:
                    seen.add(normalized_id)
                    ordered_ids.append(normalized_id)

        normalized_value = dict(value)
        normalized_value["source_block_ids"] = ordered_ids
        return normalized_value

    @field_validator("source_block_ids", mode="before")
    @classmethod
    def normalize_source_ids(cls, value: object) -> object:
        """Reject blank IDs and deduplicate section evidence IDs."""

        return _deduplicate_ids(value)

    @model_validator(mode="after")
    def validate_source_id_union(self) -> ResumeSection:
        expected_ids = _ordered_source_block_ids(self.items)
        if self.source_block_ids != expected_ids:
            raise ValueError(
                "source_block_ids must equal the ordered union of item source_block_ids."
            )
        return self


class StructuredResume(StrictModel):
    """A structured resume that retains content that could not be classified."""

    sections: list[ResumeSection]
    unclassified_content: list[ResumeItem]
    warnings: list[str]

    @field_validator("warnings", mode="before")
    @classmethod
    def clean_warnings(cls, value: object) -> object:
        """Remove blank warning entries while preserving valid messages."""

        return _clean_string_list(value)

    @model_validator(mode="after")
    def validate_content_present(self) -> StructuredResume:
        if not self.sections and not self.unclassified_content:
            raise ValueError("A structured resume must retain at least one content item.")
        return self


class JobRequirement(StrictModel):
    """A requirement explicitly extracted from a job description."""

    requirement_id: NonEmptyString
    category: RequirementCategory
    description: NonEmptyString
    importance: RequirementImportance
    source_excerpt: NonEmptyString


class JobProfile(StrictModel):
    """A structured job summary and its non-empty requirement set."""

    role_summary: NonEmptyString
    requirements: Annotated[list[JobRequirement], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_unique_requirement_ids(self) -> JobProfile:
        requirement_ids = [requirement.requirement_id for requirement in self.requirements]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("Job requirement IDs must be unique.")
        return self


class RequirementAssessment(StrictModel):
    """An evidence-aware assessment of one job requirement."""

    requirement_id: NonEmptyString
    status: AssessmentStatus
    source_block_ids: list[str]
    reason: NonEmptyString
    suggested_action: NonEmptyString

    @field_validator("source_block_ids", mode="before")
    @classmethod
    def normalize_source_ids(cls, value: object) -> object:
        """Reject blank IDs and deduplicate assessment evidence IDs."""

        return _deduplicate_ids(value)

    @model_validator(mode="after")
    def validate_evidence_for_status(self) -> RequirementAssessment:
        if self.status == "unsupported" and self.source_block_ids:
            raise ValueError("Unsupported requirements must not reference source blocks.")
        if self.status != "unsupported" and not self.source_block_ids:
            raise ValueError(f"{self.status} requirements must reference source blocks.")
        return self


class MatchAnalysis(StrictModel):
    """A qualitative, evidence-aware resume-to-job match analysis."""

    overall_rating: OverallRating
    overall_evaluation: NonEmptyString
    assessments: Annotated[list[RequirementAssessment], Field(min_length=1)]
    main_issues: list[str]
    section_suggestions: list[str]
    keyword_suggestions: list[str]
    truthfulness_risks: list[str]
    content_not_to_add: list[str]

    @field_validator(
        "main_issues",
        "section_suggestions",
        "keyword_suggestions",
        "truthfulness_risks",
        "content_not_to_add",
        mode="before",
    )
    @classmethod
    def clean_text_lists(cls, value: object) -> object:
        """Remove blank entries from report-oriented string lists."""

        return _clean_string_list(value)

    @model_validator(mode="after")
    def validate_unique_assessment_ids(self) -> MatchAnalysis:
        requirement_ids = [assessment.requirement_id for assessment in self.assessments]
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("Assessment requirement IDs must be unique.")
        return self


class OptimizedResume(StrictModel):
    """The authoritative optimized resume representation."""

    sections: Annotated[list[ResumeSection], Field(min_length=1)]
    pending_user_inputs: list[str]
    warnings: list[str]

    @field_validator("pending_user_inputs", "warnings", mode="before")
    @classmethod
    def clean_text_lists(cls, value: object) -> object:
        """Remove blank pending-input and warning entries."""

        return _clean_string_list(value)


class OptimizationResult(StrictModel):
    """The complete in-memory result of a resume optimization run."""

    analysis: MatchAnalysis
    optimized_resume: OptimizedResume
    output_paths: dict[OutputPathKey, Path]
    warnings: list[str]

    @field_validator("warnings", mode="before")
    @classmethod
    def clean_warnings(cls, value: object) -> object:
        """Remove blank warning entries while preserving valid messages."""

        return _clean_string_list(value)
