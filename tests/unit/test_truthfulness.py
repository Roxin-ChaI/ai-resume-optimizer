"""Deterministic tests for optimized-resume truthfulness checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_resume_optimizer.exceptions import TruthfulnessError
from ai_resume_optimizer.models import (
    ExtractedResume,
    JobProfile,
    JobRequirement,
    MatchAnalysis,
    OptimizedResume,
    RequirementAssessment,
    ResumeItem,
    ResumeSection,
    SourceBlock,
)
from ai_resume_optimizer.services.truthfulness import validate_optimized_resume


def _extracted_resume(tmp_path: Path, text: str) -> ExtractedResume:
    return ExtractedResume(
        source_path=tmp_path / "resume.docx",
        source_format="docx",
        blocks=[
            SourceBlock(
                block_id="block-0001",
                text=text,
                kind="paragraph",
                location="body:1",
            )
        ],
        plain_text=text,
        warnings=[],
    )


def _job_profile() -> JobProfile:
    return JobProfile(
        role_summary="Build services.",
        requirements=[
            JobRequirement(
                requirement_id="requirement-0001",
                category="core_skill",
                description="Python",
                importance="required",
                source_excerpt="Python",
            )
        ],
    )


def _match_analysis(
    *,
    requirement_id: str = "requirement-0001",
    status: str = "well_supported",
) -> MatchAnalysis:
    source_ids = [] if status == "unsupported" else ["block-0001"]
    return MatchAnalysis(
        overall_rating="一般",
        overall_evaluation="Evidence is present.",
        assessments=[
            RequirementAssessment(
                requirement_id=requirement_id,
                status=status,
                source_block_ids=source_ids,
                reason="Evidence assessment.",
                suggested_action="Use only supported evidence.",
            )
        ],
        main_issues=[],
        section_suggestions=[],
        keyword_suggestions=[],
        truthfulness_risks=[],
        content_not_to_add=[],
    )


def _optimized_resume(
    text: str,
    *,
    source_block_ids: list[str] | None = None,
    related_requirement_ids: list[str] | None = None,
    needs_review: bool = True,
    pending_user_inputs: list[str] | None = None,
) -> OptimizedResume:
    item = ResumeItem(
        text=text,
        source_block_ids=source_block_ids or ["block-0001"],
        related_requirement_ids=related_requirement_ids or [],
        needs_review=needs_review,
        review_note="Review this wording." if needs_review else None,
    )
    return OptimizedResume(
        sections=[
            ResumeSection(
                section_type="experience",
                title="Experience",
                items=[item],
                source_block_ids=item.source_block_ids,
            )
        ],
        pending_user_inputs=pending_user_inputs or [],
        warnings=[],
    )


def _validate(
    tmp_path: Path,
    source_text: str,
    optimized_text: str,
    **optimized_arguments: object,
) -> None:
    validate_optimized_resume(
        _extracted_resume(tmp_path, source_text),
        _job_profile(),
        _match_analysis(),
        _optimized_resume(optimized_text, **optimized_arguments),
    )


def test_truthfulness_accepts_unchanged_fact_and_returns_none(tmp_path: Path) -> None:
    result = _validate(
        tmp_path,
        "Built Python APIs.",
        "Built Python APIs.",
        needs_review=False,
        related_requirement_ids=["requirement-0001"],
    )

    assert result is None


def test_truthfulness_accepts_conservative_light_rewrite(tmp_path: Path) -> None:
    _validate(
        tmp_path,
        "Built Python APIs",
        "Built  Python APIs.",
        needs_review=False,
    )


def test_truthfulness_rejects_unknown_source_block_without_leaking_resume(
    tmp_path: Path,
) -> None:
    full_resume = "Confidential complete resume body"

    with pytest.raises(TruthfulnessError) as raised:
        _validate(
            tmp_path,
            full_resume,
            full_resume,
            source_block_ids=["block-9999"],
        )

    assert full_resume not in str(raised.value)


def test_truthfulness_rejects_unknown_requirement(tmp_path: Path) -> None:
    with pytest.raises(TruthfulnessError):
        _validate(
            tmp_path,
            "Python",
            "Python",
            related_requirement_ids=["requirement-9999"],
        )


def test_truthfulness_rejects_unsupported_requirement(tmp_path: Path) -> None:
    with pytest.raises(TruthfulnessError):
        validate_optimized_resume(
            _extracted_resume(tmp_path, "Python"),
            _job_profile(),
            _match_analysis(status="unsupported"),
            _optimized_resume(
                "Python",
                related_requirement_ids=["requirement-0001"],
            ),
        )


def test_truthfulness_rejects_mismatched_match_analysis(tmp_path: Path) -> None:
    with pytest.raises(TruthfulnessError):
        validate_optimized_resume(
            _extracted_resume(tmp_path, "Python"),
            _job_profile(),
            _match_analysis(requirement_id="requirement-9999"),
            _optimized_resume("Python"),
        )


@pytest.mark.parametrize(
    ("source_text", "optimized_text"),
    [
        ("Handled 20 requests.", "Handled 20 requests."),
        ("Handled 1,000 requests.", "Handled 1000 requests."),
        ("Version 01 was deployed.", "Version 1 was deployed."),
        ("Improved speed by 20%.", "Improved speed by 20%."),
        ("Used version 3.0.", "Used version 3."),
        ("Improved throughput by +15.", "Improved throughput by 15."),
        ("Reduced variance by -3.2.", "Reduced variance by -3.2."),
    ],
)
def test_truthfulness_accepts_supported_normalized_numbers(
    tmp_path: Path,
    source_text: str,
    optimized_text: str,
) -> None:
    _validate(tmp_path, source_text, optimized_text)


@pytest.mark.parametrize(
    ("source_text", "optimized_text"),
    [
        ("Handled 20 requests.", "Improved throughput by 20%."),
        ("Improved throughput.", "Improved throughput by 40."),
        ("Improved throughput.", "Improved throughput by 40%."),
        ("Handled 20 and 30 requests.", "Handled 50 requests."),
    ],
)
def test_truthfulness_rejects_unsupported_numbers(
    tmp_path: Path,
    source_text: str,
    optimized_text: str,
) -> None:
    with pytest.raises(TruthfulnessError) as raised:
        _validate(tmp_path, source_text, optimized_text)

    assert optimized_text not in str(raised.value)


@pytest.mark.parametrize(
    ("source_text", "optimized_text"),
    [
        ("Worked from 2023年1月.", "Worked from 2023-01."),
        ("Started on 2023/1/2.", "Started on 2023-01-02."),
    ],
)
def test_truthfulness_accepts_equivalent_date_formats(
    tmp_path: Path,
    source_text: str,
    optimized_text: str,
) -> None:
    _validate(tmp_path, source_text, optimized_text)


@pytest.mark.parametrize(
    ("source_text", "optimized_text"),
    [
        ("Worked in 2023.", "Worked from 2023-01."),
        ("Worked from 2023-01.", "Worked from 2023-01-15."),
        ("Worked previously.", "Worked in 2025."),
    ],
)
def test_truthfulness_rejects_unsupported_dates(
    tmp_path: Path,
    source_text: str,
    optimized_text: str,
) -> None:
    with pytest.raises(TruthfulnessError):
        _validate(tmp_path, source_text, optimized_text)


def test_truthfulness_allows_pending_content_only_in_pending_list(
    tmp_path: Path,
) -> None:
    _validate(
        tmp_path,
        "Built Python APIs.",
        "Built Python APIs.",
        pending_user_inputs=["Please confirm the team size."],
    )


def test_truthfulness_rejects_pending_text_duplicated_in_body(tmp_path: Path) -> None:
    pending = "Please confirm the team size."

    with pytest.raises(TruthfulnessError):
        _validate(
            tmp_path,
            pending,
            pending,
            pending_user_inputs=[pending],
        )


@pytest.mark.parametrize(
    "placeholder",
    [
        "TODO add result",
        "TBD result",
        "项目成果待补充",
        "[填写成果]",
        "(请填写)",
        "<insert value>",
        "<fill here>",
    ],
)
def test_truthfulness_rejects_placeholder_content(
    tmp_path: Path,
    placeholder: str,
) -> None:
    with pytest.raises(TruthfulnessError):
        _validate(tmp_path, placeholder, placeholder)


def test_truthfulness_allows_ordinary_bracketed_text(tmp_path: Path) -> None:
    _validate(tmp_path, "Used Python [asyncio].", "Used Python [asyncio].")


def test_truthfulness_does_not_require_review_for_identical_text(
    tmp_path: Path,
) -> None:
    _validate(tmp_path, "Python", "Python", needs_review=False)


def test_truthfulness_allows_minor_short_text_change_without_review(
    tmp_path: Path,
) -> None:
    _validate(tmp_path, "Python", "Python!", needs_review=False)


def test_truthfulness_requires_review_for_major_expansion(tmp_path: Path) -> None:
    with pytest.raises(TruthfulnessError):
        _validate(
            tmp_path,
            "Built APIs.",
            "Designed and led a company-wide distributed platform architecture.",
            needs_review=False,
        )


def test_truthfulness_accepts_review_marker_for_major_rewrite(tmp_path: Path) -> None:
    _validate(
        tmp_path,
        "Built APIs.",
        "Designed and led a company-wide distributed platform architecture.",
        needs_review=True,
    )


def test_truthfulness_allows_conservative_review_marker(tmp_path: Path) -> None:
    _validate(
        tmp_path,
        "Built Python APIs.",
        "Built Python APIs.",
        needs_review=True,
    )


def test_truthfulness_rewrite_threshold_is_deterministic(tmp_path: Path) -> None:
    evidence = "a" * 20
    at_boundary = evidence + ("b" * 16)
    above_boundary = evidence + ("b" * 17)

    _validate(tmp_path, evidence, at_boundary, needs_review=False)
    with pytest.raises(TruthfulnessError):
        _validate(tmp_path, evidence, above_boundary, needs_review=False)
