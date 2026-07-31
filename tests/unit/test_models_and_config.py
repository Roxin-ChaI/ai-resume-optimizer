"""Tests for stage-one settings, exceptions, and strict data models."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError

import ai_resume_optimizer.config as config_module
from ai_resume_optimizer.config import (
    DEFAULT_OPENAI_TIMEOUT_SECONDS,
    MAX_JOB_DESCRIPTION_CHARACTERS,
    MAX_RESUME_CHARACTERS,
    Settings,
    load_settings,
)
from ai_resume_optimizer.exceptions import (
    ConfigurationError,
    InputError,
    InputTooLargeError,
    ModelCallError,
    ModelOutputError,
    OutputError,
    ResumeExtractionError,
    ResumeOptimizerError,
    TruthfulnessError,
    UnsupportedFormatError,
)
from ai_resume_optimizer.models import (
    ExtractedResume,
    JobProfile,
    JobRequirement,
    MatchAnalysis,
    OptimizationResult,
    OptimizedResume,
    RequirementAssessment,
    ResumeItem,
    ResumeSection,
    SourceBlock,
    StructuredResume,
)


def make_block(block_id: str = "block-0001") -> SourceBlock:
    return SourceBlock(
        block_id=block_id,
        text="Built reliable services",
        kind="paragraph",
        location="page:1",
    )


def make_item(
    text: str = "Built reliable services",
    source_block_ids: list[str] | None = None,
) -> ResumeItem:
    return ResumeItem(
        text=text,
        source_block_ids=source_block_ids or ["block-0001"],
        related_requirement_ids=[],
        needs_review=False,
        review_note=None,
    )


def make_section() -> ResumeSection:
    return ResumeSection(
        section_type="experience",
        title="Experience",
        items=[make_item()],
        source_block_ids=["block-0001"],
    )


def make_requirement(requirement_id: str = "req-0001") -> JobRequirement:
    return JobRequirement(
        requirement_id=requirement_id,
        category="core_skill",
        description="Python",
        importance="required",
        source_excerpt="Strong Python experience",
    )


def make_assessment(
    requirement_id: str = "req-0001",
    status: str = "well_supported",
    source_block_ids: list[str] | None = None,
) -> RequirementAssessment:
    if source_block_ids is None:
        source_block_ids = [] if status == "unsupported" else ["block-0001"]
    return RequirementAssessment(
        requirement_id=requirement_id,
        status=status,
        source_block_ids=source_block_ids,
        reason="The resume contains supporting evidence.",
        suggested_action="Keep the evidence prominent.",
    )


def make_analysis(rating: str = "高") -> MatchAnalysis:
    return MatchAnalysis(
        overall_rating=rating,
        overall_evaluation="The resume is well aligned.",
        assessments=[make_assessment()],
        main_issues=[],
        section_suggestions=[],
        keyword_suggestions=[],
        truthfulness_risks=[],
        content_not_to_add=[],
    )


def make_optimized_resume() -> OptimizedResume:
    return OptimizedResume(
        sections=[make_section()],
        pending_user_inputs=[],
        warnings=[],
    )


def set_required_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")


def test_load_settings_with_complete_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12.5")

    settings = load_settings()

    assert settings == Settings(
        openai_api_key="test-api-key",
        openai_model="test-model",
        openai_timeout_seconds=12.5,
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OPENAI_API_KEY", None),
        ("OPENAI_API_KEY", "   "),
        ("OPENAI_MODEL", None),
        ("OPENAI_MODEL", "   "),
    ],
)
def test_required_settings_reject_missing_or_blank_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str | None,
) -> None:
    set_required_environment(monkeypatch)
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=name):
        load_settings()


def test_load_settings_uses_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required_environment(monkeypatch)
    monkeypatch.delenv("OPENAI_TIMEOUT_SECONDS", raising=False)

    assert load_settings().openai_timeout_seconds == DEFAULT_OPENAI_TIMEOUT_SECONDS


@pytest.mark.parametrize("value", ["not-a-number", "0", "-3", "nan", "inf"])
def test_load_settings_rejects_invalid_timeout(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    set_required_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", value)

    with pytest.raises(ConfigurationError, match="OPENAI_TIMEOUT_SECONDS"):
        load_settings()


def test_configuration_error_does_not_include_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sensitive-secret-value"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "invalid")

    with pytest.raises(ConfigurationError) as error:
        load_settings()

    assert secret not in str(error.value)


def test_importing_config_does_not_read_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    reloaded = importlib.reload(config_module)

    assert reloaded.MAX_RESUME_CHARACTERS == 50_000
    assert reloaded.MAX_JOB_DESCRIPTION_CHARACTERS == 30_000
    assert MAX_RESUME_CHARACTERS == 50_000
    assert MAX_JOB_DESCRIPTION_CHARACTERS == 30_000


def test_models_reject_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SourceBlock(
            block_id="block-0001",
            text="Text",
            kind="paragraph",
            location="page:1",
            unexpected=True,
        )


def test_models_strip_strings_and_reject_blank_required_strings() -> None:
    block = SourceBlock(
        block_id="  block-0001  ",
        text="  Text  ",
        kind="paragraph",
        location="  page:1  ",
    )

    assert block.block_id == "block-0001"
    assert block.text == "Text"
    assert block.location == "page:1"

    with pytest.raises(ValidationError):
        SourceBlock(
            block_id=" ",
            text="Text",
            kind="paragraph",
            location="page:1",
        )


def test_string_lists_remove_blank_items_and_strip_values() -> None:
    resume = ExtractedResume(
        source_path=Path("resume.pdf"),
        source_format="pdf",
        blocks=[make_block()],
        plain_text="Text",
        warnings=["  warning  ", "", "   "],
    )

    assert resume.warnings == ["warning"]


def test_id_lists_deduplicate_in_first_seen_order() -> None:
    item = ResumeItem(
        text="Text",
        source_block_ids=[" block-2 ", "block-1", "block-2"],
        related_requirement_ids=["req-2", " req-1 ", "req-2"],
        needs_review=False,
        review_note=None,
    )

    assert item.source_block_ids == ["block-2", "block-1"]
    assert item.related_requirement_ids == ["req-2", "req-1"]


def test_extracted_resume_accepts_valid_content() -> None:
    resume = ExtractedResume(
        source_path=Path("resume.pdf"),
        source_format="pdf",
        blocks=[make_block()],
        plain_text="Built reliable services",
        warnings=[],
    )

    assert resume.blocks[0].block_id == "block-0001"


@pytest.mark.parametrize(
    "updates",
    [
        {"blocks": []},
        {"plain_text": "   "},
        {"blocks": [make_block(), make_block()]},
    ],
)
def test_extracted_resume_rejects_invalid_content(updates: dict[str, object]) -> None:
    data: dict[str, object] = {
        "source_path": Path("resume.pdf"),
        "source_format": "pdf",
        "blocks": [make_block()],
        "plain_text": "Built reliable services",
        "warnings": [],
    }
    data.update(updates)

    with pytest.raises(ValidationError):
        ExtractedResume.model_validate(data)


def test_resume_item_accepts_valid_fact() -> None:
    assert make_item().source_block_ids == ["block-0001"]


def test_resume_item_rejects_blank_source_id() -> None:
    with pytest.raises(ValidationError):
        make_item(source_block_ids=[" "])


@pytest.mark.parametrize(
    ("needs_review", "review_note"),
    [(True, None), (False, "Review this wording.")],
)
def test_resume_item_rejects_inconsistent_review_state(
    needs_review: bool,
    review_note: str | None,
) -> None:
    with pytest.raises(ValidationError):
        ResumeItem(
            text="Text",
            source_block_ids=["block-0001"],
            related_requirement_ids=[],
            needs_review=needs_review,
            review_note=review_note,
        )


def test_resume_section_accepts_correct_ordered_union() -> None:
    first = make_item(source_block_ids=["block-2", "block-1"])
    second = make_item(source_block_ids=["block-1", "block-3"])
    section = ResumeSection(
        section_type="experience",
        title="Experience",
        items=[first, second],
        source_block_ids=["block-2", "block-1", "block-3"],
    )

    assert section.source_block_ids == ["block-2", "block-1", "block-3"]


def test_resume_section_rejects_empty_items() -> None:
    with pytest.raises(ValidationError):
        ResumeSection(
            section_type="experience",
            title="Experience",
            items=[],
            source_block_ids=[],
        )


def test_resume_section_rejects_incorrect_source_union() -> None:
    with pytest.raises(ValidationError):
        ResumeSection(
            section_type="experience",
            title="Experience",
            items=[make_item()],
            source_block_ids=["block-other"],
        )


def test_structured_resume_accepts_sections_or_unclassified_content() -> None:
    with_section = StructuredResume(
        sections=[make_section()],
        unclassified_content=[],
        warnings=[],
    )
    with_unclassified = StructuredResume(
        sections=[],
        unclassified_content=[make_item()],
        warnings=[],
    )

    assert with_section.sections
    assert with_unclassified.unclassified_content


def test_structured_resume_rejects_no_content() -> None:
    with pytest.raises(ValidationError):
        StructuredResume(sections=[], unclassified_content=[], warnings=[])


def test_job_profile_accepts_valid_requirements() -> None:
    profile = JobProfile(role_summary="Backend role", requirements=[make_requirement()])

    assert profile.requirements[0].requirement_id == "req-0001"


def test_job_profile_rejects_empty_or_duplicate_requirements() -> None:
    with pytest.raises(ValidationError):
        JobProfile(role_summary="Backend role", requirements=[])

    with pytest.raises(ValidationError):
        JobProfile(
            role_summary="Backend role",
            requirements=[make_requirement(), make_requirement()],
        )


@pytest.mark.parametrize(
    ("status", "source_block_ids", "is_valid"),
    [
        ("well_supported", ["block-0001"], True),
        ("well_supported", [], False),
        ("underrepresented", ["block-0001"], True),
        ("underrepresented", [], False),
        ("unsupported", [], True),
        ("unsupported", ["block-0001"], False),
    ],
)
def test_requirement_assessment_evidence_rules(
    status: str,
    source_block_ids: list[str],
    is_valid: bool,
) -> None:
    if is_valid:
        assessment = make_assessment(status=status, source_block_ids=source_block_ids)
        assert assessment.status == status
    else:
        with pytest.raises(ValidationError):
            make_assessment(status=status, source_block_ids=source_block_ids)


@pytest.mark.parametrize("rating", ["高", "一般", "低"])
def test_match_analysis_accepts_approved_ratings(rating: str) -> None:
    assert make_analysis(rating).overall_rating == rating


def test_match_analysis_rejects_unknown_rating() -> None:
    with pytest.raises(ValidationError):
        make_analysis("优秀")


def test_match_analysis_rejects_empty_or_duplicate_assessments() -> None:
    base = make_analysis().model_dump()
    base["assessments"] = []
    with pytest.raises(ValidationError):
        MatchAnalysis.model_validate(base)

    base["assessments"] = [make_assessment(), make_assessment()]
    with pytest.raises(ValidationError):
        MatchAnalysis.model_validate(base)


def test_optimized_resume_requires_sections_and_separates_pending_inputs() -> None:
    optimized = OptimizedResume(
        sections=[make_section()],
        pending_user_inputs=["  Add a verified metric.  ", ""],
        warnings=[],
    )

    assert optimized.pending_user_inputs == ["Add a verified metric."]
    assert all(
        item.text != optimized.pending_user_inputs[0] for item in optimized.sections[0].items
    )

    with pytest.raises(ValidationError):
        OptimizedResume(sections=[], pending_user_inputs=[], warnings=[])


def test_optimization_result_accepts_empty_or_approved_output_paths() -> None:
    empty_paths = OptimizationResult(
        analysis=make_analysis(),
        optimized_resume=make_optimized_resume(),
        output_paths={},
        warnings=[],
    )
    all_paths = OptimizationResult(
        analysis=make_analysis(),
        optimized_resume=make_optimized_resume(),
        output_paths={
            "analysis_report": Path("output/analysis_report.md"),
            "optimized_resume_markdown": Path("output/optimized_resume.md"),
            "optimized_resume_docx": Path("output/optimized_resume.docx"),
        },
        warnings=[],
    )

    assert empty_paths.output_paths == {}
    assert len(all_paths.output_paths) == 3


def test_optimization_result_rejects_unapproved_output_path_key() -> None:
    with pytest.raises(ValidationError):
        OptimizationResult(
            analysis=make_analysis(),
            optimized_resume=make_optimized_resume(),
            output_paths={"pdf_export": Path("output/resume.pdf")},
            warnings=[],
        )


@pytest.mark.parametrize(
    ("exception_type", "exit_code"),
    [
        (ResumeOptimizerError, 1),
        (ConfigurationError, 4),
        (InputError, 2),
        (UnsupportedFormatError, 2),
        (InputTooLargeError, 2),
        (ResumeExtractionError, 3),
        (ModelCallError, 4),
        (ModelOutputError, 5),
        (TruthfulnessError, 5),
        (OutputError, 6),
    ],
)
def test_exception_hierarchy_and_exit_codes(
    exception_type: type[ResumeOptimizerError],
    exit_code: int,
) -> None:
    error = exception_type("safe message")

    assert isinstance(error, ResumeOptimizerError)
    assert error.exit_code == exit_code
    assert str(error) == "safe message"
