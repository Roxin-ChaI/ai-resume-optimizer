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
    MatchAnalysis,
    OptimizedResume,
    RequirementAssessment,
    ResumeItem,
    ResumeSection,
    SourceBlock,
    StructuredResume,
)
from ai_resume_optimizer.prompts import load_prompt
from ai_resume_optimizer.services.analysis import (
    analyze_job,
    analyze_match,
    optimize_resume,
    structure_resume,
)
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


def _extracted_resume_with_blocks(
    tmp_path: Path,
    blocks: list[SourceBlock],
) -> ExtractedResume:
    return ExtractedResume(
        source_path=tmp_path / "resume.docx",
        source_format="docx",
        blocks=blocks,
        plain_text="\n".join(block.text for block in blocks),
        warnings=[],
    )


def _parse_structure_resume_input(input_text: str) -> tuple[int, list[str], dict[str, object]]:
    manifest_text, source_text = input_text.split("SOURCE_BLOCKS:\n", maxsplit=1)
    manifest_lines = manifest_text.splitlines()
    count = int(manifest_lines[0].removeprefix("REQUIRED_SOURCE_BLOCK_COUNT: "))
    assert manifest_lines[1] == "REQUIRED_SOURCE_BLOCK_IDS:"
    required_ids = json.loads(manifest_lines[2])
    source_data = json.loads(source_text)
    return count, required_ids, source_data


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


def _match_analysis(
    assessments: list[RequirementAssessment] | None = None,
    *,
    overall_rating: str = "一般",
    overall_evaluation: str = "The resume has relevant evidence with some gaps.",
) -> MatchAnalysis:
    return MatchAnalysis(
        overall_rating=overall_rating,
        overall_evaluation=overall_evaluation,
        assessments=assessments
        or [
            RequirementAssessment(
                requirement_id="requirement-0001",
                status="well_supported",
                source_block_ids=["block-0002"],
                reason="Python is listed.",
                suggested_action="Keep the evidence visible.",
            ),
            RequirementAssessment(
                requirement_id="requirement-0002",
                status="unsupported",
                source_block_ids=[],
                reason="No SQL evidence is present.",
                suggested_action="Do not add SQL without evidence.",
            ),
        ],
        main_issues=[],
        section_suggestions=[],
        keyword_suggestions=[],
        truthfulness_risks=[],
        content_not_to_add=["SQL"],
    )


def _optimized_resume(
    item: ResumeItem | None = None,
    *,
    pending_user_inputs: list[str] | None = None,
) -> OptimizedResume:
    resume_item = item or _resume_item(
        "Python",
        ["block-0002"],
        related_requirement_ids=["requirement-0001"],
    )
    return OptimizedResume(
        sections=[
            ResumeSection(
                section_type="skills",
                title="Skills",
                items=[resume_item],
                source_block_ids=resume_item.source_block_ids,
            )
        ],
        pending_user_inputs=pending_user_inputs or [],
        warnings=[],
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
    count, required_ids, payload = _parse_structure_resume_input(call.input_text)
    assert count == 2
    assert required_ids == ["block-0001", "block-0002"]
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


def test_structure_resume_manifest_preserves_every_block_in_original_order(
    tmp_path: Path,
) -> None:
    blocks = [
        SourceBlock(
            block_id="block-0001",
            text="Profile summary",
            kind="paragraph",
            location="body:1",
        ),
        SourceBlock(
            block_id="block-0002",
            text="技能概览",
            kind="paragraph",
            location="body:2",
        ),
        SourceBlock(
            block_id="block-0003",
            text="Example table row",
            kind="table_row",
            location="table:1,row:1",
        ),
    ]
    extracted = _extracted_resume_with_blocks(tmp_path, blocks)
    expected = StructuredResume(
        sections=[],
        unclassified_content=[_resume_item(block.text, [block.block_id]) for block in blocks],
        warnings=[],
    )
    client = FakeModelClient({StructuredResume: expected})

    structure_resume(extracted, client)

    assert len(client.calls) == 1
    input_text = client.calls[0].input_text
    count, required_ids, source_data = _parse_structure_resume_input(input_text)
    assert count == len(blocks)
    assert required_ids == [block.block_id for block in blocks]
    assert required_ids[1] == "block-0002"
    assert source_data == {
        "source_format": "docx",
        "blocks": [block.model_dump(mode="json") for block in blocks],
    }
    manifest_text = input_text.split("SOURCE_BLOCKS:\n", maxsplit=1)[0]
    assert all(block.text not in manifest_text for block in blocks)


def test_structure_resume_prompt_requires_exact_manifest_coverage(tmp_path: Path) -> None:
    client = FakeModelClient({StructuredResume: _structured_resume()})

    structure_resume(_extracted_resume(tmp_path), client)

    assert len(client.calls) == 1
    instructions = client.calls[0].instructions
    for required_phrase in (
        "sole authoritative and complete set",
        "must be exactly equal",
        "no required ID is omitted",
        "no ID outside the manifest",
        "combines multiple source blocks",
        "every combined block ID",
        "short content, including section headings",
        "short single-line paragraphs",
        "unclassified_content",
    ):
        assert required_phrase in instructions


def test_structure_resume_accepts_covered_four_character_heading(tmp_path: Path) -> None:
    blocks = [
        SourceBlock(
            block_id="block-0001",
            text="技能概览",
            kind="paragraph",
            location="body:1",
        )
    ]
    extracted = _extracted_resume_with_blocks(tmp_path, blocks)
    expected = StructuredResume(
        sections=[],
        unclassified_content=[_resume_item("技能概览", ["block-0001"])],
        warnings=[],
    )

    actual = structure_resume(extracted, FakeModelClient({StructuredResume: expected}))

    assert actual is expected


def test_structure_resume_rejects_omitted_four_character_heading_once(
    tmp_path: Path,
) -> None:
    heading_text = "技能概览"
    blocks = [
        SourceBlock(
            block_id="block-0001",
            text=heading_text,
            kind="paragraph",
            location="body:1",
        ),
        SourceBlock(
            block_id="block-0002",
            text="Example detail",
            kind="paragraph",
            location="body:2",
        ),
    ]
    result = StructuredResume(
        sections=[],
        unclassified_content=[_resume_item("Example detail", ["block-0002"])],
        warnings=[],
    )
    client = FakeModelClient({StructuredResume: result})

    with pytest.raises(ModelOutputError, match="block-0001") as raised:
        structure_resume(_extracted_resume_with_blocks(tmp_path, blocks), client)

    assert heading_text not in str(raised.value)
    assert len(client.calls) == 1


@pytest.mark.parametrize("covered_ids", [["block-0001", "block-0002"], ["block-0001"]])
def test_structure_resume_merged_item_must_cover_every_block(
    covered_ids: list[str],
    tmp_path: Path,
) -> None:
    blocks = [
        SourceBlock(
            block_id="block-0001",
            text="First detail",
            kind="paragraph",
            location="body:1",
        ),
        SourceBlock(
            block_id="block-0002",
            text="Second detail",
            kind="paragraph",
            location="body:2",
        ),
    ]
    result = StructuredResume(
        sections=[],
        unclassified_content=[_resume_item("Combined detail", covered_ids)],
        warnings=[],
    )
    client = FakeModelClient({StructuredResume: result})

    if len(covered_ids) == len(blocks):
        assert structure_resume(_extracted_resume_with_blocks(tmp_path, blocks), client) is result
    else:
        with pytest.raises(ModelOutputError, match="block-0002"):
            structure_resume(_extracted_resume_with_blocks(tmp_path, blocks), client)
    assert len(client.calls) == 1


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
    client = FakeModelClient({StructuredResume: result})

    with pytest.raises(ModelOutputError) as raised:
        structure_resume(extracted, client)

    assert extracted.plain_text not in str(raised.value)
    assert len(client.calls) == 1


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
    client = FakeModelClient({StructuredResume: result})

    with pytest.raises(ModelOutputError, match="block-9999"):
        structure_resume(_extracted_resume(tmp_path), client)
    assert len(client.calls) == 1


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


def test_analyze_match_sends_ordered_safe_json_and_returns_result(
    tmp_path: Path,
) -> None:
    structured_resume = _structured_resume()
    job_profile = _job_profile()
    expected = _match_analysis()
    before_resume = structured_resume.model_dump()
    before_job = job_profile.model_dump()
    client = FakeModelClient({MatchAnalysis: expected})

    actual = analyze_match(client, structured_resume, job_profile)

    assert actual is expected
    assert structured_resume.model_dump() == before_resume
    assert job_profile.model_dump() == before_job
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call.response_model is MatchAnalysis
    assert call.instructions == load_prompt("analyze_match.txt")
    payload = json.loads(call.input_text)
    assert list(payload) == ["structured_resume", "job_profile"]
    assert [section["title"] for section in payload["structured_resume"]["sections"]] == ["Skills"]
    assert [
        requirement["requirement_id"] for requirement in payload["job_profile"]["requirements"]
    ] == ["requirement-0001", "requirement-0002"]
    assert str(tmp_path) not in call.input_text


@pytest.mark.parametrize(
    "assessments",
    [
        [
            RequirementAssessment(
                requirement_id="requirement-0001",
                status="well_supported",
                source_block_ids=["block-0002"],
                reason="Evidence exists.",
                suggested_action="Keep it.",
            )
        ],
        [
            RequirementAssessment(
                requirement_id="requirement-0001",
                status="well_supported",
                source_block_ids=["block-0002"],
                reason="Evidence exists.",
                suggested_action="Keep it.",
            ),
            RequirementAssessment(
                requirement_id="requirement-0002",
                status="unsupported",
                source_block_ids=[],
                reason="No evidence.",
                suggested_action="Do not add it.",
            ),
            RequirementAssessment(
                requirement_id="requirement-9999",
                status="unsupported",
                source_block_ids=[],
                reason="Unknown requirement.",
                suggested_action="Do not add it.",
            ),
        ],
        [
            RequirementAssessment(
                requirement_id="requirement-0002",
                status="unsupported",
                source_block_ids=[],
                reason="No evidence.",
                suggested_action="Do not add it.",
            ),
            RequirementAssessment(
                requirement_id="requirement-0001",
                status="well_supported",
                source_block_ids=["block-0002"],
                reason="Evidence exists.",
                suggested_action="Keep it.",
            ),
        ],
    ],
)
def test_analyze_match_rejects_missing_unknown_or_reordered_assessments(
    assessments: list[RequirementAssessment],
) -> None:
    client = FakeModelClient({MatchAnalysis: _match_analysis(assessments)})

    with pytest.raises(ModelOutputError):
        analyze_match(client, _structured_resume(), _job_profile())

    assert [
        assessment.requirement_id for assessment in client.responses[MatchAnalysis].assessments
    ] == [assessment.requirement_id for assessment in assessments]


def test_analyze_match_rejects_duplicate_assessment_ids() -> None:
    duplicate = _match_analysis().model_construct(
        assessments=[
            RequirementAssessment(
                requirement_id="requirement-0001",
                status="well_supported",
                source_block_ids=["block-0002"],
                reason="Evidence exists.",
                suggested_action="Keep it.",
            ),
            RequirementAssessment(
                requirement_id="requirement-0001",
                status="well_supported",
                source_block_ids=["block-0002"],
                reason="Evidence exists.",
                suggested_action="Keep it.",
            ),
        ]
    )

    with pytest.raises(ModelOutputError):
        analyze_match(
            FakeModelClient({MatchAnalysis: duplicate}),
            _structured_resume(),
            _job_profile(),
        )


def test_analyze_match_accepts_section_and_unclassified_block_evidence() -> None:
    structured_resume = StructuredResume(
        sections=[
            ResumeSection(
                section_type="skills",
                title="Skills",
                items=[_resume_item("Python", ["block-0001"])],
                source_block_ids=["block-0001"],
            )
        ],
        unclassified_content=[_resume_item("SQL", ["block-0002"])],
        warnings=[],
    )
    assessments = [
        RequirementAssessment(
            requirement_id="requirement-0001",
            status="well_supported",
            source_block_ids=["block-0001"],
            reason="Section evidence.",
            suggested_action="Keep it.",
        ),
        RequirementAssessment(
            requirement_id="requirement-0002",
            status="underrepresented",
            source_block_ids=["block-0002"],
            reason="Unclassified evidence.",
            suggested_action="Clarify it.",
        ),
    ]
    expected = _match_analysis(assessments)

    assert (
        analyze_match(
            FakeModelClient({MatchAnalysis: expected}),
            structured_resume,
            _job_profile(),
        )
        is expected
    )


def test_analyze_match_rejects_unknown_assessment_block_without_leaking_resume() -> None:
    resume = _structured_resume()
    assessments = [
        RequirementAssessment(
            requirement_id="requirement-0001",
            status="well_supported",
            source_block_ids=["block-9999"],
            reason="Invalid evidence.",
            suggested_action="Keep it.",
        ),
        _match_analysis().assessments[1],
    ]

    with pytest.raises(ModelOutputError) as raised:
        analyze_match(
            FakeModelClient({MatchAnalysis: _match_analysis(assessments)}),
            resume,
            _job_profile(),
        )

    assert resume.model_dump_json() not in str(raised.value)


@pytest.mark.parametrize("rating", ["高", "一般", "低"])
def test_analyze_match_accepts_all_qualitative_ratings(rating: str) -> None:
    expected = _match_analysis(overall_rating=rating)

    assert (
        analyze_match(
            FakeModelClient({MatchAnalysis: expected}),
            _structured_resume(),
            _job_profile(),
        )
        is expected
    )


@pytest.mark.parametrize(
    "evaluation",
    [
        "Internal match rating: 80%",
        "内部匹配评价80%",
        "ATS score is strong.",
        "评价包含ATS score。",
        "ATS 分数较高。",
        "预计通过率较高。",
        "预计录取率较高。",
        "招聘平台评分较高。",
    ],
)
def test_analyze_match_rejects_numeric_scores_and_recruiting_predictions(
    evaluation: str,
) -> None:
    with pytest.raises(ModelOutputError):
        analyze_match(
            FakeModelClient({MatchAnalysis: _match_analysis(overall_evaluation=evaluation)}),
            _structured_resume(),
            _job_profile(),
        )


def test_analyze_match_allows_ats_as_an_ordinary_skill_term() -> None:
    expected = _match_analysis(
        overall_evaluation="The resume shows experience administering ATS software."
    )

    assert (
        analyze_match(
            FakeModelClient({MatchAnalysis: expected}),
            _structured_resume(),
            _job_profile(),
        )
        is expected
    )


def test_optimize_resume_sends_ordered_safe_json_and_returns_result(
    tmp_path: Path,
) -> None:
    structured_resume = _structured_resume()
    job_profile = _job_profile()
    match_analysis = _match_analysis()
    expected = _optimized_resume()
    before = (
        structured_resume.model_dump(),
        job_profile.model_dump(),
        match_analysis.model_dump(),
    )
    client = FakeModelClient({OptimizedResume: expected})

    actual = optimize_resume(
        client,
        structured_resume,
        job_profile,
        match_analysis,
    )

    assert actual is expected
    assert before == (
        structured_resume.model_dump(),
        job_profile.model_dump(),
        match_analysis.model_dump(),
    )
    call = client.calls[0]
    assert len(client.calls) == 1
    assert call.response_model is OptimizedResume
    assert call.instructions == load_prompt("optimize_resume.txt")
    payload = json.loads(call.input_text)
    assert list(payload) == [
        "structured_resume",
        "job_profile",
        "match_analysis",
    ]
    assert [
        requirement["requirement_id"] for requirement in payload["job_profile"]["requirements"]
    ] == ["requirement-0001", "requirement-0002"]
    assert [section["title"] for section in payload["structured_resume"]["sections"]] == ["Skills"]
    assert str(tmp_path) not in call.input_text


def test_optimize_resume_accepts_valid_source_and_requirement_ids() -> None:
    expected = _optimized_resume()

    assert (
        optimize_resume(
            FakeModelClient({OptimizedResume: expected}),
            _structured_resume(),
            _job_profile(),
            _match_analysis(),
        )
        is expected
    )


def test_optimize_resume_rejects_unknown_source_block_id() -> None:
    item = _resume_item("Python", ["block-9999"])
    expected = _optimized_resume(item)

    with pytest.raises(ModelOutputError):
        optimize_resume(
            FakeModelClient({OptimizedResume: expected}),
            _structured_resume(),
            _job_profile(),
            _match_analysis(),
        )


def test_optimize_resume_rejects_unknown_requirement_id() -> None:
    item = _resume_item(
        "Python",
        ["block-0002"],
        related_requirement_ids=["requirement-9999"],
    )

    with pytest.raises(ModelOutputError):
        optimize_resume(
            FakeModelClient({OptimizedResume: _optimized_resume(item)}),
            _structured_resume(),
            _job_profile(),
            _match_analysis(),
        )


def test_optimize_resume_rejects_unsupported_requirement_reference() -> None:
    item = _resume_item(
        "Python",
        ["block-0002"],
        related_requirement_ids=["requirement-0002"],
    )

    with pytest.raises(ModelOutputError):
        optimize_resume(
            FakeModelClient({OptimizedResume: _optimized_resume(item)}),
            _structured_resume(),
            _job_profile(),
            _match_analysis(),
        )


def test_optimize_resume_rejects_mismatched_analysis_before_model_call() -> None:
    mismatched = _match_analysis([_match_analysis().assessments[0]])
    client = FakeModelClient({OptimizedResume: _optimized_resume()})

    with pytest.raises(ModelOutputError):
        optimize_resume(
            client,
            _structured_resume(),
            _job_profile(),
            mismatched,
        )

    assert client.calls == []


@pytest.mark.parametrize(
    "error",
    [
        ModelCallError("provider unavailable"),
        ModelOutputError("invalid provider output"),
    ],
)
def test_optimize_resume_propagates_model_errors(error: Exception) -> None:
    client = FakeModelClient(error=error)

    with pytest.raises(type(error)) as raised:
        optimize_resume(
            client,
            _structured_resume(),
            _job_profile(),
            _match_analysis(),
        )

    assert raised.value is error
