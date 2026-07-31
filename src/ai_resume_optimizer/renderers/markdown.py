"""Deterministic Markdown renderers for analysis and optimized resumes."""

from __future__ import annotations

from ai_resume_optimizer.models import MatchAnalysis, OptimizedResume, ResumeItem

_STATUS_LABELS = {
    "well_supported": "Well Supported",
    "underrepresented": "Underrepresented",
    "unsupported": "Unsupported",
}
_BULLETED_SECTION_TYPES = {
    "skills",
    "experience",
    "projects",
    "education",
    "certifications",
    "other",
}
_IMPORTANT_NOTICE = (
    "This report is an assistive analysis. It is not an ATS score or hiring "
    "prediction, and it does not guarantee any outcome. Review all content "
    "carefully before use."
)


def _finish(lines: list[str]) -> str:
    return "\n".join(lines).rstrip("\n") + "\n"


def _append_string_list(lines: list[str], values: list[str]) -> None:
    if values:
        lines.extend(f"- {value}" for value in values)
    else:
        lines.append("None identified.")


def render_analysis_report(analysis: MatchAnalysis) -> str:
    """Render a stable Markdown analysis report."""

    lines = [
        "# AI Resume Optimizer Analysis Report",
        "",
        "## Overall Match",
        "",
        f"**Rating:** {analysis.overall_rating}",
        "",
        analysis.overall_evaluation,
        "",
        "## Requirement Assessments",
        "",
    ]
    for assessment in analysis.assessments:
        source_ids = ", ".join(assessment.source_block_ids) or "None"
        lines.extend(
            [
                f"### {assessment.requirement_id}",
                "",
                f"- **Status:** {_STATUS_LABELS[assessment.status]}",
                f"- **Reason:** {assessment.reason}",
                f"- **Suggested Action:** {assessment.suggested_action}",
                f"- **Source Block IDs:** {source_ids}",
                "",
            ]
        )

    sections = [
        ("Main Issues", analysis.main_issues),
        ("Section Suggestions", analysis.section_suggestions),
        ("Keyword Suggestions", analysis.keyword_suggestions),
        ("Truthfulness Risks", analysis.truthfulness_risks),
        ("Content Not to Add", analysis.content_not_to_add),
    ]
    for title, values in sections:
        lines.extend([f"## {title}", ""])
        _append_string_list(lines, values)
        lines.append("")

    lines.extend(["## Important Notice", "", _IMPORTANT_NOTICE])
    return _finish(lines)


def _markdown_item(item: ResumeItem, *, bulleted: bool) -> list[str]:
    item_lines = item.text.splitlines()
    if not bulleted:
        return item_lines
    return [f"- {item_lines[0]}", *(f"  {line}" for line in item_lines[1:])]


def render_optimized_resume(resume: OptimizedResume) -> str:
    """Render a stable Markdown resume from the authoritative model."""

    lines = ["# Optimized Resume", ""]
    review_notes: list[str] = []
    for section in resume.sections:
        lines.extend([f"## {section.title}", ""])
        bulleted = section.section_type in _BULLETED_SECTION_TYPES
        for item_number, item in enumerate(section.items, start=1):
            lines.extend(_markdown_item(item, bulleted=bulleted))
            lines.append("")
            if item.needs_review:
                review_notes.append(f"- {section.title}, item {item_number}: {item.review_note}")

    if review_notes:
        lines.extend(["## Review Notes", "", *review_notes, ""])
    if resume.pending_user_inputs:
        lines.extend(["## Pending User Inputs", ""])
        _append_string_list(lines, resume.pending_user_inputs)
        lines.append("")
    if resume.warnings:
        lines.extend(["## Warnings", ""])
        _append_string_list(lines, resume.warnings)

    return _finish(lines)
