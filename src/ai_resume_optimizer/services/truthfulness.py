"""Deterministic truthfulness checks for optimized resume content."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from difflib import SequenceMatcher

from ai_resume_optimizer.exceptions import TruthfulnessError
from ai_resume_optimizer.models import (
    ExtractedResume,
    JobProfile,
    MatchAnalysis,
    OptimizedResume,
    ResumeItem,
    SourceBlock,
)

_NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w)")
_SEPARATED_DATE_PATTERN = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})(?P<separator>[-/.])"
    r"(?P<month>\d{1,2})(?:(?P=separator)(?P<day>\d{1,2}))?(?!\d)"
)
_CHINESE_DATE_PATTERN = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})年(?P<month>\d{1,2})月"
    r"(?:(?P<day>\d{1,2})日)?"
)
_YEAR_PATTERN = re.compile(r"(?<!\d)(?P<year>(?:19|20)\d{2})(?!\d)")
_INLINE_PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:TODO|TBD)\b|待补充|请填写|请补充|待确认",
    re.IGNORECASE,
)
_ANGLE_PLACEHOLDER_PATTERN = re.compile(
    r"<\s*(?:insert|fill)\b[^>]*>",
    re.IGNORECASE,
)
_WRAPPED_PLACEHOLDER_PATTERN = re.compile(
    r"^\s*[\[(（]\s*"
    r"(?=[^\])）]*(?:TODO|TBD|填写|补充|确认|insert|fill))"
    r"[^\])）]*[\])）]\s*$",
    re.IGNORECASE,
)

_SIGNIFICANT_REWRITE_RATIO = 0.55
_SHORT_REWRITE_RATIO = 0.35
_MAX_EXPANSION_RATIO = 1.8
_SHORT_MAX_EXPANSION_RATIO = 2.5
_MAX_COMPRESSION_RATIO = 2.5
_SHORT_TEXT_LENGTH = 20


@dataclass(frozen=True)
class _DateMatch:
    value: tuple[int, int | None, int | None]
    span: tuple[int, int]


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _iter_resume_items(optimized_resume: OptimizedResume) -> list[ResumeItem]:
    return [item for section in optimized_resume.sections for item in section.items]


def _extract_dates(text: str) -> list[_DateMatch]:
    matches: list[_DateMatch] = []
    occupied: list[tuple[int, int]] = []

    for pattern in (_CHINESE_DATE_PATTERN, _SEPARATED_DATE_PATTERN, _YEAR_PATTERN):
        for match in pattern.finditer(text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            month_text = match.groupdict().get("month")
            day_text = match.groupdict().get("day")
            matches.append(
                _DateMatch(
                    value=(
                        int(match.group("year")),
                        int(month_text) if month_text is not None else None,
                        int(day_text) if day_text is not None else None,
                    ),
                    span=span,
                )
            )
            occupied.append(span)

    return sorted(matches, key=lambda item: item.span)


def _inside_spans(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def _normalize_number(value: str) -> tuple[Decimal, bool]:
    is_percentage = value.endswith("%")
    number = value[:-1] if is_percentage else value
    number = number.replace(",", "")
    if number.startswith("+"):
        number = number[1:]
    return Decimal(number), is_percentage


def _extract_numbers(
    text: str,
    date_matches: list[_DateMatch],
) -> set[tuple[Decimal, bool]]:
    date_spans = [match.span for match in date_matches]
    return {
        _normalize_number(match.group())
        for match in _NUMBER_PATTERN.finditer(text)
        if not _inside_spans(match.start(), date_spans)
    }


def _validate_item_facts(
    item: ResumeItem,
    blocks_by_id: dict[str, SourceBlock],
    item_number: int,
) -> None:
    evidence_blocks: list[SourceBlock] = []
    for block_id in item.source_block_ids:
        block = blocks_by_id.get(block_id)
        if block is None:
            raise TruthfulnessError(
                f"Optimized resume item {item_number} references an unknown source block."
            )
        evidence_blocks.append(block)

    evidence_text = "\n".join(block.text for block in evidence_blocks)
    evidence_dates = _extract_dates(evidence_text)
    optimized_dates = _extract_dates(item.text)
    evidence_date_values = {match.value for match in evidence_dates}
    if any(match.value not in evidence_date_values for match in optimized_dates):
        raise TruthfulnessError(
            f"Optimized resume item {item_number} contains an unsupported date."
        )

    evidence_numbers = _extract_numbers(evidence_text, evidence_dates)
    optimized_numbers = _extract_numbers(item.text, optimized_dates)
    if not optimized_numbers.issubset(evidence_numbers):
        raise TruthfulnessError(
            f"Optimized resume item {item_number} contains an unsupported number."
        )

    if _is_significant_rewrite(evidence_text, item.text) and not item.needs_review:
        raise TruthfulnessError(f"Optimized resume item {item_number} requires human review.")


def _is_significant_rewrite(evidence_text: str, optimized_text: str) -> bool:
    evidence = _normalized_text(evidence_text)
    optimized = _normalized_text(optimized_text)
    if evidence == optimized:
        return False
    if not evidence or not optimized:
        return True

    similarity = SequenceMatcher(None, evidence, optimized).ratio()
    shorter_length = min(len(evidence), len(optimized))
    longer_length = max(len(evidence), len(optimized))
    if longer_length < _SHORT_TEXT_LENGTH:
        return (
            similarity < _SHORT_REWRITE_RATIO
            or len(optimized) > len(evidence) * _SHORT_MAX_EXPANSION_RATIO
        )
    if similarity < _SIGNIFICANT_REWRITE_RATIO:
        return True
    if len(optimized) > len(evidence) * _MAX_EXPANSION_RATIO:
        return True
    return (
        len(evidence) > len(optimized) * _MAX_COMPRESSION_RATIO
        and optimized not in evidence
        and shorter_length > 0
    )


def _validate_requirement_relationships(
    job_profile: JobProfile,
    match_analysis: MatchAnalysis,
    optimized_resume: OptimizedResume,
) -> None:
    requirement_ids = [requirement.requirement_id for requirement in job_profile.requirements]
    assessment_ids = [assessment.requirement_id for assessment in match_analysis.assessments]
    if (
        len(assessment_ids) != len(requirement_ids)
        or len(assessment_ids) != len(set(assessment_ids))
        or set(assessment_ids) != set(requirement_ids)
    ):
        raise TruthfulnessError(
            "Match assessments do not correspond exactly to the job requirements."
        )

    valid_requirement_ids = set(requirement_ids)
    unsupported_ids = {
        assessment.requirement_id
        for assessment in match_analysis.assessments
        if assessment.status == "unsupported"
    }
    for item_number, item in enumerate(_iter_resume_items(optimized_resume), start=1):
        for requirement_id in item.related_requirement_ids:
            if requirement_id not in valid_requirement_ids:
                raise TruthfulnessError(
                    f"Optimized resume item {item_number} references an unknown requirement."
                )
            if requirement_id in unsupported_ids:
                raise TruthfulnessError(
                    f"Optimized resume item {item_number} references an unsupported requirement."
                )


def _validate_pending_input_separation(optimized_resume: OptimizedResume) -> None:
    pending_inputs = {_normalized_text(value) for value in optimized_resume.pending_user_inputs}
    for item_number, item in enumerate(_iter_resume_items(optimized_resume), start=1):
        normalized_item = _normalized_text(item.text)
        if normalized_item in pending_inputs:
            raise TruthfulnessError(
                f"Optimized resume item {item_number} duplicates a pending user input."
            )
        if (
            _INLINE_PLACEHOLDER_PATTERN.search(item.text)
            or _ANGLE_PLACEHOLDER_PATTERN.search(item.text)
            or _WRAPPED_PLACEHOLDER_PATTERN.fullmatch(item.text)
        ):
            raise TruthfulnessError(
                f"Optimized resume item {item_number} contains placeholder content."
            )


def validate_optimized_resume(
    extracted_resume: ExtractedResume,
    job_profile: JobProfile,
    match_analysis: MatchAnalysis,
    optimized_resume: OptimizedResume,
) -> None:
    """Block deterministic evidence and truthfulness violations."""

    _validate_requirement_relationships(
        job_profile,
        match_analysis,
        optimized_resume,
    )
    _validate_pending_input_separation(optimized_resume)

    blocks_by_id = {block.block_id: block for block in extracted_resume.blocks}
    for item_number, item in enumerate(_iter_resume_items(optimized_resume), start=1):
        _validate_item_facts(item, blocks_by_id, item_number)
