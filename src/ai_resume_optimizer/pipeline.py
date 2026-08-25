"""Single-run resume optimization pipeline and atomic output handling."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile

from ai_resume_optimizer.exceptions import ModelOutputError, OutputError
from ai_resume_optimizer.model_client import ModelClient
from ai_resume_optimizer.models import (
    EvidenceSectionReference,
    ExtractedResume,
    JobProfile,
    MatchAnalysis,
    OptimizationResult,
    OptimizedResume,
    RequirementEvidence,
    RequirementReference,
    StructuredResume,
)
from ai_resume_optimizer.parsers import parse_resume
from ai_resume_optimizer.parsers.job_description import normalize_job_description
from ai_resume_optimizer.renderers import (
    render_analysis_report_markdown,
    render_optimized_resume_docx,
    render_optimized_resume_markdown,
)
from ai_resume_optimizer.services.analysis import (
    analyze_job,
    analyze_match,
    optimize_resume,
    structure_resume,
)
from ai_resume_optimizer.services.truthfulness import validate_optimized_resume

_OUTPUT_FILENAMES = {
    "analysis_report": "analysis_report.md",
    "optimized_resume_markdown": "optimized_resume.md",
    "optimized_resume_docx": "optimized_resume.docx",
}


def _prepare_output_paths(output_dir: Path) -> dict[str, Path]:
    try:
        if output_dir.exists():
            if not output_dir.is_dir():
                raise OutputError(f"Output path '{output_dir}' is not a directory.")
        else:
            output_dir.mkdir(parents=True)
    except OutputError:
        raise
    except OSError as error:
        raise OutputError(f"Could not create output directory '{output_dir}'.") from error

    paths = {key: output_dir / filename for key, filename in _OUTPUT_FILENAMES.items()}
    try:
        existing = [path for path in paths.values() if path.exists()]
    except OSError as error:
        raise OutputError(f"Could not inspect output directory '{output_dir}'.") from error
    if existing:
        raise OutputError(f"Refusing to overwrite existing output file '{existing[0]}'.")
    return paths


def _render_outputs(
    analysis: MatchAnalysis,
    optimized_resume: OptimizedResume,
) -> tuple[bytes, bytes, bytes]:
    try:
        analysis_markdown = render_analysis_report_markdown(analysis)
        resume_markdown = render_optimized_resume_markdown(optimized_resume)
        resume_docx = render_optimized_resume_docx(optimized_resume)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise OutputError("Could not render the requested output files.") from error
    return (
        analysis_markdown.encode("utf-8"),
        resume_markdown.encode("utf-8"),
        resume_docx,
    )


def _write_temporary_file(output_dir: Path, final_name: str, content: bytes) -> Path:
    temporary = NamedTemporaryFile(
        mode="wb",
        dir=output_dir,
        prefix=f".{final_name}.",
        suffix=".tmp",
        delete=False,
    )
    path = Path(temporary.name)
    try:
        with temporary:
            temporary.write(content)
    except OSError as error:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            raise error
        raise
    return path


def _replace_file(source: Path, target: Path) -> None:
    source.replace(target)


def _cleanup_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


def _write_output_batch(
    output_paths: dict[str, Path],
    rendered_outputs: tuple[bytes, bytes, bytes],
) -> None:
    ordered_paths = [
        output_paths["analysis_report"],
        output_paths["optimized_resume_markdown"],
        output_paths["optimized_resume_docx"],
    ]
    temporary_paths: list[Path] = []
    created_final_paths: list[Path] = []
    try:
        for target, content in zip(ordered_paths, rendered_outputs, strict=True):
            temporary_paths.append(_write_temporary_file(target.parent, target.name, content))
        for temporary_path, target in zip(
            temporary_paths,
            ordered_paths,
            strict=True,
        ):
            if target.exists():
                raise FileExistsError(f"Output file appeared during processing: {target}")
            _replace_file(temporary_path, target)
            created_final_paths.append(target)
    except OSError as error:
        _cleanup_paths(temporary_paths)
        _cleanup_paths(created_final_paths)
        raise OutputError("Could not write the complete output file batch.") from error


def _merge_warnings(*warning_groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for warning in (item for group in warning_groups for item in group):
        if warning and warning not in seen:
            seen.add(warning)
            merged.append(warning)
    return merged


def _attach_requirement_provenance(
    match_analysis: MatchAnalysis,
    *,
    job_profile: JobProfile,
    extracted_resume: ExtractedResume,
    structured_resume: StructuredResume,
) -> MatchAnalysis:
    """Attach deterministic job and source references to every assessment."""

    requirements_by_id = {
        requirement.requirement_id: requirement for requirement in job_profile.requirements
    }
    source_blocks_by_id = {
        source_block.block_id: source_block for source_block in extracted_resume.blocks
    }
    sections_by_source_block_id: dict[str, list[EvidenceSectionReference]] = {}
    for section in structured_resume.sections:
        reference = EvidenceSectionReference(
            section_type=section.section_type,
            title=section.title,
        )
        for source_block_id in section.source_block_ids:
            sections_by_source_block_id.setdefault(source_block_id, []).append(reference)

    enriched_assessments = []
    for assessment in match_analysis.assessments:
        requirement = requirements_by_id.get(assessment.requirement_id)
        if requirement is None:
            raise ModelOutputError(
                f"Match assessment references unknown job requirement "
                f"{assessment.requirement_id!r}."
            )

        evidence = []
        for source_block_id in assessment.source_block_ids:
            source_block = source_blocks_by_id.get(source_block_id)
            if source_block is None:
                raise ModelOutputError(
                    f"Match assessment {assessment.requirement_id!r} references "
                    f"unknown source block ID {source_block_id!r}."
                )
            evidence.append(
                RequirementEvidence(
                    source_block_id=source_block.block_id,
                    kind=source_block.kind,
                    location=source_block.location,
                    excerpt=source_block.text,
                    sections=list(sections_by_source_block_id.get(source_block_id, [])),
                )
            )

        enriched_assessments.append(
            assessment.model_copy(
                update={
                    "requirement": RequirementReference(
                        requirement_id=requirement.requirement_id,
                        description=requirement.description,
                        category=requirement.category,
                        importance=requirement.importance,
                        source_excerpt=requirement.source_excerpt,
                    ),
                    "evidence": evidence,
                }
            )
        )

    return match_analysis.model_copy(update={"assessments": enriched_assessments})


def _run_optimization_in_memory(
    *,
    resume_path: Path,
    job_description: str,
    model_client: ModelClient,
) -> OptimizationResult:
    """Run the optimization workflow without rendering or writing outputs."""

    normalized_job_description = normalize_job_description(job_description)
    extracted_resume = parse_resume(resume_path)
    structured_resume = structure_resume(extracted_resume, model_client)
    job_profile = analyze_job(normalized_job_description, model_client)
    match_analysis = analyze_match(
        model_client,
        structured_resume,
        job_profile,
    )
    optimized_resume = optimize_resume(
        model_client,
        structured_resume,
        job_profile,
        match_analysis,
    )
    validate_optimized_resume(
        extracted_resume,
        job_profile,
        match_analysis,
        optimized_resume,
    )
    enriched_match_analysis = _attach_requirement_provenance(
        match_analysis,
        job_profile=job_profile,
        extracted_resume=extracted_resume,
        structured_resume=structured_resume,
    )

    return OptimizationResult(
        analysis=enriched_match_analysis,
        optimized_resume=optimized_resume,
        output_paths={},
        warnings=_merge_warnings(
            extracted_resume.warnings,
            optimized_resume.warnings,
        ),
    )


def _export_optimization_result(
    result: OptimizationResult,
    output_paths: dict[str, Path],
) -> OptimizationResult:
    """Render and atomically write one in-memory optimization result."""

    rendered_outputs = _render_outputs(result.analysis, result.optimized_resume)
    _write_output_batch(output_paths, rendered_outputs)
    return OptimizationResult(
        analysis=result.analysis,
        optimized_resume=result.optimized_resume,
        output_paths=output_paths,
        warnings=result.warnings,
    )


def run_optimization(
    *,
    resume_path: Path,
    job_description: str,
    output_dir: Path,
    model_client: ModelClient,
) -> OptimizationResult:
    """Run one complete optimization and atomically write its three outputs."""

    output_paths = _prepare_output_paths(output_dir)
    result = _run_optimization_in_memory(
        resume_path=resume_path,
        job_description=job_description,
        model_client=model_client,
    )
    return _export_optimization_result(result, output_paths)
