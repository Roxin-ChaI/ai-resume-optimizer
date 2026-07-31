"""Offline tests for structured resume and job analysis services."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_resume_optimizer.exceptions import ModelCallError, ModelOutputError
from ai_resume_optimizer.models import (
    ExtractedResume,
    JobProfile,
    JobRequirement,
    ResumeItem,
    ResumeSection,
    SourceBlock,
    StructuredResume,
)
from ai_resume_optimizer.prompts import load_prompt
from ai_resume_optimizer.services.analysis import analyze_job, structure_resume
from tests.fakes import FakeModelClient


def _extracted_resume(tmp_path: Path) -> ExtractedResume:
    return ExtractedResume(
        source_path=tmp_path / "resume.docx",
        source_format="docx",
        blocks=[
            SourceBlock(
                block_id="block-0001",
                text="Skills",
                kind="heading",
                location="body:1",
            ),
            SourceBlock(
                block_id="block-0002",
                text="Python",
                kind="list_item",
                location="body:2",
            ),
        ],
        plain_text="Skills\nPython",
        warnings=[],
    )


def _resume_item(
    text: str,
    block_ids: list[str],
    *,
    related_requirement_ids: list[str] | None = None,
    needs_review: bool = False,
    review_note: str | None = None,
) -> ResumeItem:
    return ResumeItem(
        text=text,
        source_block_ids=block_ids,
        related_requirement_ids=related_requirement_ids or [],
        needs_review=needs_review,
        review_note=review_note,
    )


def _structured_resume() -> StructuredResume:
    items = [
        _resume_item("Skills", ["block-0001"]),
        _resume_item("Python", ["block-0002"]),
    ]
    return StructuredResume(
        sections=[
            ResumeSection(
                section_type="skills",
                title="Skills",
                items=items,
                source_block_ids=["block-0001", "block-0002"],
            )
        ],
        unclassified_content=[],
        warnings=[],
    )


def _job_profile(
    requirements: list[JobRequirement] | None = None,
) -> JobProfile:
    return JobProfile(
        role_summary="Build Python services.",
        requirements=requirements
        or [
            JobRequirement(
                requirement_id="requirement-0001",
                category="core_skill",
                description="Python",
                importance="required",
                source_excerpt="Python",
            ),
            JobRequirement(
                requirement_id="requirement-0002",
                category="preferred_skill",
                description="SQL",
                importance="preferred",
                source_excerpt="SQL is preferred",
            ),
        ],
    )


@pytest.mark.parametrize("name", ["structure_resume.txt", "analyze_job.txt"])
def test_load_prompt_returns_approved_nonempty_resource(name: str) -> None:
    assert load_prompt(name).strip()


def test_load_prompt_rejects_unknown_resource() -> None:
    with pytest.raises(ValueError):
        load_prompt("../secret.txt")


def test_load_prompt_is_independent_of_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert "StructuredResume" in load_prompt("structure_resume.txt")


def test_structure_resume_sends_ordered_safe_json_and_returns_result(
    tmp_path: Path,
) -> None:
    extracted = _extracted_resume(tmp_path)
    before = extracted.model_dump()
    expected = _structured_resume()
    client = FakeModelClient({StructuredResume: expected})

    actual = structure_resume(extracted, client)

    assert actual is expected
    assert extracted.model_dump() == before
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call.instructions == load_prompt("structure_resume.txt")
    assert call.response_model is StructuredResume
    payload = json.loads(call.input_text)
    assert payload == {
        "source_format": "docx",
        "blocks": [
            {
                "block_id": "block-0001",
                "kind": "heading",
                "location": "body:1",
                "text": "Skills",
            },
            {
                "block_id": "block-0002",
                "kind": "list_item",
                "location": "body:2",
                "text": "Python",
            },
        ],
    }
    assert str(extracted.source_path) not in call.input_text


def test_structure_resume_accepts_unclassified_only_with_complete_coverage(
    tmp_path: Path,
) -> None:
    expected = StructuredResume(
        sections=[],
        unclassified_content=[
            _resume_item("Skills", ["block-0001"]),
            _resume_item("Python", ["block-0002"]),
        ],
        warnings=[],
    )

    assert (
        structure_resume(
            _extracted_resume(tmp_path),
            FakeModelClient({StructuredResume: expected}),
        )
        is expected
    )


@pytest.mark.parametrize(
    "result",
    [
        StructuredResume(
            sections=[],
            unclassified_content=[_resume_item("Unknown", ["block-9999"])],
            warnings=[],
        ),
        StructuredResume(
            sections=[],
            unclassified_content=[_resume_item("Skills", ["block-0001"])],
            warnings=[],
        ),
        StructuredResume(
            sections=[
                ResumeSection(
                    section_type="skills",
                    title="Skills",
                    items=[
                        _resume_item(
                            "Skills",
                            ["block-0001"],
                            related_requirement_ids=["requirement-0001"],
                        ),
                        _resume_item("Python", ["block-0002"]),
                    ],
                    source_block_ids=["block-0001", "block-0002"],
                )
            ],
            unclassified_content=[],
            warnings=[],
        ),
        StructuredResume(
            sections=[],
            unclassified_content=[
                _resume_item(
                    "Skills",
                    ["block-0001"],
                    related_requirement_ids=["requirement-0001"],
                ),
                _resume_item("Python", ["block-0002"]),
            ],
            warnings=[],
        ),
        StructuredResume(
            sections=[
                ResumeSection(
                    section_type="skills",
                    title="Skills",
                    items=[
                        _resume_item(
                            "Skills",
                            ["block-0001"],
                            needs_review=True,
                            review_note="Review",
                        ),
                        _resume_item("Python", ["block-0002"]),
                    ],
                    source_block_ids=["block-0001", "block-0002"],
                )
            ],
            unclassified_content=[],
            warnings=[],
        ),
        StructuredResume(
            sections=[],
            unclassified_content=[
                _resume_item(
                    "Skills",
                    ["block-0001"],
                    needs_review=True,
                    review_note="Review",
                ),
                _resume_item("Python", ["block-0002"]),
            ],
            warnings=[],
        ),
    ],
)
def test_structure_resume_rejects_cross_model_constraint_violations(
    result: StructuredResume,
    tmp_path: Path,
) -> None:
    extracted = _extracted_resume(tmp_path)

    with pytest.raises(ModelOutputError) as raised:
        structure_resume(extracted, FakeModelClient({StructuredResume: result}))

    assert extracted.plain_text not in str(raised.value)


def test_structure_resume_validates_section_aggregate_ids_too(tmp_path: Path) -> None:
    result = StructuredResume.model_construct(
        sections=[
            ResumeSection.model_construct(
                section_type="skills",
                title="Skills",
                items=[
                    _resume_item("Skills", ["block-0001"]),
                    _resume_item("Python", ["block-0002"]),
                ],
                source_block_ids=["block-9999"],
            )
        ],
        unclassified_content=[],
        warnings=[],
    )

    with pytest.raises(ModelOutputError, match="block-9999"):
        structure_resume(
            _extracted_resume(tmp_path),
            FakeModelClient({StructuredResume: result}),
        )


def test_analyze_job_sends_normalized_text_and_returns_result() -> None:
    description = "\n Python is required.\nSQL is preferred. \n"
    expected = _job_profile()
    client = FakeModelClient({JobProfile: expected})

    actual = analyze_job(description, client)

    assert actual is expected
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call.instructions == load_prompt("analyze_job.txt")
    assert call.input_text == "Python is required.\nSQL is preferred."
    assert call.response_model is JobProfile
    assert [item.requirement_id for item in actual.requirements] == [
        "requirement-0001",
        "requirement-0002",
    ]


def test_analyze_job_accepts_source_excerpt_with_whitespace_variation() -> None:
    description = "Experience with Python,\r\n  SQL, and APIs is required."
    expected = _job_profile(
        [
            JobRequirement(
                requirement_id="requirement-0001",
                category="experience",
                description="Python, SQL, and APIs",
                importance="required",
                source_excerpt="Python,\nSQL, and APIs",
            )
        ]
    )

    assert (
        analyze_job(
            description,
            FakeModelClient({JobProfile: expected}),
        )
        is expected
    )


@pytest.mark.parametrize(
    "requirements",
    [
        [
            JobRequirement(
                requirement_id="requirement-0002",
                category="core_skill",
                description="Python",
                importance="required",
                source_excerpt="Python",
            )
        ],
        [
            JobRequirement(
                requirement_id="requirement-0001",
                category="core_skill",
                description="Python",
                importance="required",
                source_excerpt="Python",
            ),
            JobRequirement(
                requirement_id="requirement-0003",
                category="preferred_skill",
                description="SQL",
                importance="preferred",
                source_excerpt="SQL",
            ),
        ],
        [
            JobRequirement(
                requirement_id="req-0001",
                category="core_skill",
                description="Python",
                importance="required",
                source_excerpt="Python",
            )
        ],
        [
            JobRequirement(
                requirement_id="requirement-0001",
                category="core_skill",
                description="Python",
                importance="required",
                source_excerpt="rewritten requirement",
            )
        ],
        [
            JobRequirement(
                requirement_id="requirement-0001",
                category="core_skill",
                description="Python",
                importance="required",
                source_excerpt="Python and Kubernetes",
            )
        ],
    ],
)
def test_analyze_job_rejects_invalid_ids_or_nonliteral_excerpts(
    requirements: list[JobRequirement],
) -> None:
    description = "Private full job description: Python and SQL are required."

    with pytest.raises(ModelOutputError) as raised:
        analyze_job(
            description,
            FakeModelClient({JobProfile: _job_profile(requirements)}),
        )

    assert any(requirement.requirement_id in str(raised.value) for requirement in requirements)
    assert description not in str(raised.value)


@pytest.mark.parametrize("description", ["", " \r\n\t"])
def test_analyze_job_rejects_blank_description(description: str) -> None:
    with pytest.raises(ValueError):
        analyze_job(description, FakeModelClient())


def test_analyze_job_rejects_non_string_description() -> None:
    with pytest.raises(ValueError):
        analyze_job(None, FakeModelClient())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "error",
    [
        ModelCallError("provider unavailable"),
        ModelOutputError("invalid provider output"),
    ],
)
def test_services_propagate_model_client_errors(
    error: Exception,
    tmp_path: Path,
) -> None:
    client = FakeModelClient(error=error)

    with pytest.raises(type(error)) as resume_raised:
        structure_resume(_extracted_resume(tmp_path), client)
    assert resume_raised.value is error

    with pytest.raises(type(error)) as job_raised:
        analyze_job("Python is required.", client)
    assert job_raised.value is error
