"""Offline tests for the Typer command-line boundary."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import TextIO

import pytest
from typer.testing import CliRunner

import ai_resume_optimizer.cli as cli_module
from ai_resume_optimizer.cli import app
from ai_resume_optimizer.config import Settings
from ai_resume_optimizer.exceptions import (
    ConfigurationError,
    InputError,
    ModelCallError,
    ModelOutputError,
    OutputError,
    ResumeExtractionError,
    TruthfulnessError,
)
from ai_resume_optimizer.models import (
    MatchAnalysis,
    OptimizationResult,
    OptimizedResume,
    RequirementAssessment,
    ResumeItem,
    ResumeSection,
)

runner = CliRunner()


class _InteractiveStream(StringIO):
    def isatty(self) -> bool:
        return True


class _NonInteractiveStream(StringIO):
    def isatty(self) -> bool:
        return False


def _analysis() -> MatchAnalysis:
    return MatchAnalysis(
        overall_rating="一般",
        overall_evaluation="Evidence is present.",
        assessments=[
            RequirementAssessment(
                requirement_id="requirement-0001",
                status="well_supported",
                source_block_ids=["block-0001"],
                reason="Evidence exists.",
                suggested_action="Keep it visible.",
            )
        ],
        main_issues=[],
        section_suggestions=[],
        keyword_suggestions=[],
        truthfulness_risks=[],
        content_not_to_add=[],
    )


def _optimized_resume() -> OptimizedResume:
    item = ResumeItem(
        text="Built Python APIs.",
        source_block_ids=["block-0001"],
        related_requirement_ids=["requirement-0001"],
        needs_review=False,
        review_note=None,
    )
    return OptimizedResume(
        sections=[
            ResumeSection(
                section_type="experience",
                title="Experience",
                items=[item],
                source_block_ids=["block-0001"],
            )
        ],
        pending_user_inputs=[],
        warnings=[],
    )


def _result(output_dir: Path, *, warnings: list[str] | None = None) -> OptimizationResult:
    return OptimizationResult(
        analysis=_analysis(),
        optimized_resume=_optimized_resume(),
        output_paths={
            "analysis_report": output_dir / "analysis_report.md",
            "optimized_resume_markdown": output_dir / "optimized_resume.md",
            "optimized_resume_docx": output_dir / "optimized_resume.docx",
        },
        warnings=warnings or [],
    )


def _install_success_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    job_text: str = "Normalized job description.",
    warnings: list[str] | None = None,
) -> dict[str, object]:
    calls: dict[str, object] = {}
    settings = Settings(
        deepseek_api_key="invalid-test-key",
        deepseek_model="deepseek-v4-flash",
        deepseek_timeout_seconds=12.0,
    )
    model_client = object()

    def read_job(path: Path) -> str:
        calls["job_path"] = path
        return job_text

    def load() -> Settings:
        calls["settings_loaded"] = True
        return settings

    def create_client(**kwargs: object) -> object:
        calls["client_kwargs"] = kwargs
        return model_client

    def run(**kwargs: object) -> OptimizationResult:
        calls["pipeline_kwargs"] = kwargs
        output_dir = kwargs["output_dir"]
        assert isinstance(output_dir, Path)
        return _result(output_dir, warnings=warnings)

    monkeypatch.setattr(cli_module, "read_job_description", read_job)
    monkeypatch.setattr(cli_module, "load_settings", load)
    monkeypatch.setattr(cli_module, "DeepSeekModelClient", create_client)
    monkeypatch.setattr(cli_module, "run_optimization", run)
    calls["tmp_path"] = tmp_path
    return calls


def test_help_commands_succeed_without_loading_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded() -> Settings:
        raise AssertionError("Help must not load settings.")

    monkeypatch.setattr(cli_module, "load_settings", fail_if_loaded)

    root_help = runner.invoke(app, ["--help"])
    optimize_help = runner.invoke(app, ["optimize", "--help"])

    assert root_help.exit_code == 0
    assert optimize_help.exit_code == 0
    assert "optimize" in root_help.output
    for option in ["--resume", "--job-description", "--output-dir"]:
        assert option in optimize_help.output
    for forbidden in ["--overwrite", "--api-key", "--ats-score"]:
        assert forbidden not in optimize_help.output


def test_cli_txt_input_loads_settings_builds_client_and_runs_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_success_dependencies(
        monkeypatch,
        tmp_path,
        warnings=["Review warning."],
    )
    resume_path = tmp_path / "resume.docx"
    job_path = tmp_path / "job.txt"
    output_dir = tmp_path / "custom-output"

    result = runner.invoke(
        app,
        [
            "optimize",
            "--resume",
            str(resume_path),
            "--job-description",
            str(job_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert calls["job_path"] == job_path
    assert calls["settings_loaded"] is True
    assert calls["client_kwargs"] == {
        "api_key": "invalid-test-key",
        "model": "deepseek-v4-flash",
        "timeout_seconds": 12.0,
    }
    pipeline_kwargs = calls["pipeline_kwargs"]
    assert isinstance(pipeline_kwargs, dict)
    assert pipeline_kwargs == {
        "resume_path": resume_path,
        "job_description": "Normalized job description.",
        "output_dir": output_dir,
        "model_client": pipeline_kwargs["model_client"],
    }
    assert "analysis_report.md" in result.output
    assert "optimized_resume.md" in result.output
    assert "optimized_resume.docx" in result.output
    assert "Warnings: 1" in result.output
    assert "Review all generated content before use." in result.output
    assert "invalid-test-key" not in result.output
    assert "Normalized job description." not in result.output


def test_cli_uses_default_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_success_dependencies(monkeypatch, tmp_path)

    result = runner.invoke(
        app,
        [
            "optimize",
            "--resume",
            str(tmp_path / "resume.docx"),
            "--job-description",
            str(tmp_path / "job.txt"),
        ],
    )

    assert result.exit_code == 0
    pipeline_kwargs = calls["pipeline_kwargs"]
    assert isinstance(pipeline_kwargs, dict)
    assert pipeline_kwargs["output_dir"] == Path("output")


def test_cli_interactive_input_stops_only_on_standalone_trimmed_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_success_dependencies(monkeypatch, tmp_path)
    stream: TextIO = _InteractiveStream(
        "Backend role\nBackend END-to-END Engineer\n END \nignored line\n"
    )
    normalized_inputs: list[str] = []

    def normalize(value: str) -> str:
        normalized_inputs.append(value)
        return "Normalized interactive job."

    monkeypatch.setattr(cli_module, "_stdin", lambda: stream)
    monkeypatch.setattr(cli_module, "normalize_job_description", normalize)

    result = runner.invoke(
        app,
        ["optimize", "--resume", str(tmp_path / "resume.docx")],
    )

    assert result.exit_code == 0
    assert normalized_inputs == ["Backend role\nBackend END-to-END Engineer"]
    pipeline_kwargs = calls["pipeline_kwargs"]
    assert isinstance(pipeline_kwargs, dict)
    assert pipeline_kwargs["job_description"] == "Normalized interactive job."
    assert "ignored line" not in str(pipeline_kwargs)


def test_cli_interactive_input_accepts_eof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_success_dependencies(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli_module,
        "_stdin",
        lambda: _InteractiveStream("Line one\nLine two"),
    )

    result = runner.invoke(
        app,
        ["optimize", "--resume", str(tmp_path / "resume.docx")],
    )

    assert result.exit_code == 0
    pipeline_kwargs = calls["pipeline_kwargs"]
    assert isinstance(pipeline_kwargs, dict)
    assert pipeline_kwargs["job_description"] == "Line one\nLine two"


def test_cli_interactive_blank_content_exits_as_input_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "_stdin",
        lambda: _InteractiveStream(" \n\t\nEND\n"),
    )

    result = runner.invoke(
        app,
        ["optimize", "--resume", str(tmp_path / "resume.docx")],
    )

    assert result.exit_code == 2
    assert "Job description is empty" in result.output
    assert "Traceback" not in result.output


def test_cli_refuses_noninteractive_stdin_without_consuming_or_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _NonInteractiveStream("piped job description")
    pipeline_called = False

    def run(**kwargs: object) -> OptimizationResult:
        nonlocal pipeline_called
        pipeline_called = True
        return _result(tmp_path)

    monkeypatch.setattr(cli_module, "_stdin", lambda: stream)
    monkeypatch.setattr(cli_module, "run_optimization", run)

    result = runner.invoke(
        app,
        ["optimize", "--resume", str(tmp_path / "resume.docx")],
    )

    assert result.exit_code == 2
    assert stream.tell() == 0
    assert pipeline_called is False


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [
        (InputError("input failed"), 2),
        (ResumeExtractionError("parse failed"), 3),
        (ConfigurationError("configuration failed"), 4),
        (ModelCallError("model call failed"), 4),
        (ModelOutputError("model output failed"), 5),
        (TruthfulnessError("truthfulness failed"), 5),
        (OutputError("output failed"), 6),
    ],
)
def test_cli_expected_errors_use_project_exit_codes_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    exit_code: int,
) -> None:
    def fail(**kwargs: object) -> None:
        raise error

    monkeypatch.setattr(cli_module, "_execute_optimization", fail)

    result = runner.invoke(
        app,
        [
            "optimize",
            "--resume",
            str(tmp_path / "resume.docx"),
            "--job-description",
            str(tmp_path / "job.txt"),
        ],
    )

    assert result.exit_code == exit_code
    assert str(error) in result.output
    assert "Traceback" not in result.output


def test_cli_unexpected_error_is_generic_and_exit_code_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_detail = "sensitive internal provider detail"

    def fail(**kwargs: object) -> None:
        raise RuntimeError(sensitive_detail)

    monkeypatch.setattr(cli_module, "_execute_optimization", fail)

    result = runner.invoke(
        app,
        [
            "optimize",
            "--resume",
            str(tmp_path / "resume.docx"),
            "--job-description",
            str(tmp_path / "job.txt"),
        ],
    )

    assert result.exit_code == 1
    assert "unexpected internal error" in result.output
    assert sensitive_detail not in result.output
    assert "Traceback" not in result.output
