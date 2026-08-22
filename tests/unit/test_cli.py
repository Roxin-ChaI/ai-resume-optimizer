"""Offline tests for the Typer command-line boundary."""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from typing import TextIO

import pytest
from typer.testing import CliRunner

import ai_resume_optimizer.cli as cli_module
from ai_resume_optimizer import ResumeOptimizerConfig
from ai_resume_optimizer.cli import app
from ai_resume_optimizer.config import Settings
from ai_resume_optimizer.exceptions import (
    ConfigurationError,
    InputError,
    InputTooLargeError,
    ModelCallError,
    ModelOutputError,
    OutputError,
    ResumeExtractionError,
    TruthfulnessError,
    UnsupportedFormatError,
)
from ai_resume_optimizer.models import (
    MatchAnalysis,
    OptimizationResult,
    OptimizedResume,
    RequirementAssessment,
    ResumeItem,
    ResumeSection,
)

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

runner = CliRunner()


class _InteractiveStream(StringIO):
    def isatty(self) -> bool:
        return True


class _NonInteractiveStream(StringIO):
    def isatty(self) -> bool:
        return False


class _FakeOptimizer:
    def __init__(
        self,
        result: OptimizationResult,
        *,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.optimize_calls: list[dict[str, object]] = []
        self.close_calls = 0

    def optimize(self, **kwargs: object) -> OptimizationResult:
        self.optimize_calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result

    def close(self) -> None:
        self.close_calls += 1


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


def _result(
    output_dir: Path | None = None,
    *,
    warnings: list[str] | None = None,
) -> OptimizationResult:
    return OptimizationResult(
        analysis=_analysis(),
        optimized_resume=_optimized_resume(),
        output_paths=(
            {
                "analysis_report": output_dir / "analysis_report.md",
                "optimized_resume_markdown": output_dir / "optimized_resume.md",
                "optimized_resume_docx": output_dir / "optimized_resume.docx",
            }
            if output_dir is not None
            else {}
        ),
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
    optimizer = _FakeOptimizer(_result(warnings=warnings))

    def read_job(path: Path) -> str:
        calls["job_path"] = path
        return job_text

    def load() -> Settings:
        calls["settings_loaded"] = True
        return settings

    def create_optimizer(config: object) -> _FakeOptimizer:
        calls["optimizer_config"] = config
        calls["factory_calls"] = int(calls.get("factory_calls", 0)) + 1
        return optimizer

    def prepare_paths(output_dir: Path) -> dict[str, Path]:
        calls["prepared_output_dir"] = output_dir
        return {
            "analysis_report": output_dir / "analysis_report.md",
            "optimized_resume_markdown": output_dir / "optimized_resume.md",
            "optimized_resume_docx": output_dir / "optimized_resume.docx",
        }

    def export_result(
        result: OptimizationResult,
        output_paths: dict[str, Path],
    ) -> OptimizationResult:
        calls["export_calls"] = int(calls.get("export_calls", 0)) + 1
        calls["export_result"] = result
        calls["export_paths"] = output_paths
        export_error = calls.get("export_error")
        if isinstance(export_error, Exception):
            raise export_error
        output_dir = output_paths["analysis_report"].parent
        return _result(output_dir, warnings=warnings)

    monkeypatch.setattr(cli_module, "read_job_description", read_job)
    monkeypatch.setattr(cli_module, "load_settings", load)
    monkeypatch.setattr(cli_module, "create_resume_optimizer", create_optimizer)
    monkeypatch.setattr(cli_module, "_prepare_output_paths", prepare_paths)
    monkeypatch.setattr(cli_module, "_export_optimization_result", export_result)
    calls["optimizer"] = optimizer
    calls["tmp_path"] = tmp_path
    return calls


def test_help_commands_succeed_without_loading_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded() -> Settings:
        raise AssertionError("Help must not load settings.")

    monkeypatch.setattr(cli_module, "load_settings", fail_if_loaded)

    monkeypatch.setenv("FORCE_COLOR", "1")

    root_help = runner.invoke(app, ["--help"])
    optimize_help = runner.invoke(app, ["optimize", "--help"])

    root_output = ANSI_ESCAPE_RE.sub("", root_help.output)
    optimize_output = ANSI_ESCAPE_RE.sub("", optimize_help.output)

    assert root_help.exit_code == 0
    assert optimize_help.exit_code == 0
    assert "optimize" in root_output
    for option in ["--resume", "--job-description", "--output-dir"]:
        assert option in optimize_output
    for forbidden in ["--overwrite", "--api-key", "--ats-score"]:
        assert forbidden not in optimize_help.output


def test_cli_txt_input_uses_public_runner_and_shared_export_once(
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
    config = calls["optimizer_config"]
    assert isinstance(config, ResumeOptimizerConfig)
    assert config.deepseek_api_key == "invalid-test-key"
    assert config.deepseek_model == "deepseek-v4-flash"
    assert config.deepseek_timeout_seconds == 12.0
    optimizer = calls["optimizer"]
    assert isinstance(optimizer, _FakeOptimizer)
    assert optimizer.optimize_calls == [
        {
            "resume_path": resume_path,
            "job_description": "Normalized job description.",
        }
    ]
    assert optimizer.close_calls == 1
    assert calls["factory_calls"] == 1
    assert calls["prepared_output_dir"] == output_dir
    assert calls["export_calls"] == 1
    assert calls["export_result"] is optimizer.result
    assert "analysis_report.md" in result.output
    assert "optimized_resume.md" in result.output
    assert "optimized_resume.docx" in result.output
    assert "Warnings: 1" in result.output
    assert "Review all generated content before use." in result.output
    assert "invalid-test-key" not in result.output
    assert "Normalized job description." not in result.output


def test_cli_writes_original_three_output_files_through_shared_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        deepseek_api_key="invalid-test-key",
        deepseek_model="deepseek-v4-flash",
        deepseek_timeout_seconds=12.0,
    )
    optimizer = _FakeOptimizer(_result(warnings=["Review warning."]))
    job_path = tmp_path / "job.txt"
    job_path.write_text("Python is required.", encoding="utf-8")
    output_dir = tmp_path / "output"

    monkeypatch.setattr(cli_module, "load_settings", lambda: settings)
    monkeypatch.setattr(
        cli_module,
        "create_resume_optimizer",
        lambda config: optimizer,
    )

    result = runner.invoke(
        app,
        [
            "optimize",
            "--resume",
            str(tmp_path / "resume.docx"),
            "--job-description",
            str(job_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "analysis_report.md",
        "optimized_resume.docx",
        "optimized_resume.md",
    ]
    assert (output_dir / "analysis_report.md").read_text(encoding="utf-8")
    assert (output_dir / "optimized_resume.md").read_text(encoding="utf-8")
    assert (output_dir / "optimized_resume.docx").read_bytes()
    assert len(optimizer.optimize_calls) == 1
    assert optimizer.close_calls == 1


def test_cli_refuses_overwrite_before_optimize_and_closes_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        deepseek_api_key="invalid-test-key",
        deepseek_model="deepseek-v4-flash",
        deepseek_timeout_seconds=12.0,
    )
    optimizer = _FakeOptimizer(_result())
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing = output_dir / "optimized_resume.md"
    existing.write_text("existing user content", encoding="utf-8")

    monkeypatch.setattr(cli_module, "read_job_description", lambda path: "Python role")
    monkeypatch.setattr(cli_module, "load_settings", lambda: settings)
    monkeypatch.setattr(
        cli_module,
        "create_resume_optimizer",
        lambda config: optimizer,
    )

    result = runner.invoke(
        app,
        [
            "optimize",
            "--resume",
            str(tmp_path / "resume.docx"),
            "--job-description",
            str(tmp_path / "job.txt"),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 6
    assert "Refusing to overwrite" in result.output
    assert optimizer.optimize_calls == []
    assert optimizer.close_calls == 1
    assert existing.read_text(encoding="utf-8") == "existing user content"


def test_cli_optimization_error_skips_export_and_closes_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_success_dependencies(monkeypatch, tmp_path)
    optimizer = calls["optimizer"]
    assert isinstance(optimizer, _FakeOptimizer)
    optimizer.error = ModelCallError("model call failed")

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

    assert result.exit_code == 4
    assert "model call failed" in result.output
    assert len(optimizer.optimize_calls) == 1
    assert optimizer.close_calls == 1
    assert "export_calls" not in calls


def test_cli_export_error_closes_runner_and_keeps_output_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_success_dependencies(monkeypatch, tmp_path)
    calls["export_error"] = OutputError("output failed")
    optimizer = calls["optimizer"]
    assert isinstance(optimizer, _FakeOptimizer)

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

    assert result.exit_code == 6
    assert "output failed" in result.output
    assert len(optimizer.optimize_calls) == 1
    assert optimizer.close_calls == 1
    assert calls["export_calls"] == 1


def test_cli_configuration_error_prevents_factory_and_keeps_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_called = False

    def fail_load() -> Settings:
        raise ConfigurationError("DEEPSEEK_API_KEY is required and must not be blank.")

    def create_optimizer(config: object) -> _FakeOptimizer:
        nonlocal factory_called
        factory_called = True
        return _FakeOptimizer(_result())

    monkeypatch.setattr(cli_module, "read_job_description", lambda path: "Python role")
    monkeypatch.setattr(cli_module, "load_settings", fail_load)
    monkeypatch.setattr(cli_module, "create_resume_optimizer", create_optimizer)

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

    assert result.exit_code == 4
    assert "DEEPSEEK_API_KEY is required" in result.output
    assert factory_called is False


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
    assert calls["prepared_output_dir"] == Path("output")


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
    optimizer = calls["optimizer"]
    assert isinstance(optimizer, _FakeOptimizer)
    assert optimizer.optimize_calls[0]["job_description"] == "Normalized interactive job."
    assert "ignored line" not in str(optimizer.optimize_calls)


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
    optimizer = calls["optimizer"]
    assert isinstance(optimizer, _FakeOptimizer)
    assert optimizer.optimize_calls[0]["job_description"] == "Line one\nLine two"


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
    factory_called = False

    def create_optimizer(config: object) -> _FakeOptimizer:
        nonlocal factory_called
        factory_called = True
        return _FakeOptimizer(_result())

    monkeypatch.setattr(cli_module, "_stdin", lambda: stream)
    monkeypatch.setattr(cli_module, "create_resume_optimizer", create_optimizer)

    result = runner.invoke(
        app,
        ["optimize", "--resume", str(tmp_path / "resume.docx")],
    )

    assert result.exit_code == 2
    assert stream.tell() == 0
    assert factory_called is False


@pytest.mark.parametrize(
    ("error", "exit_code"),
    [
        (InputError("input failed"), 2),
        (UnsupportedFormatError("unsupported format"), 2),
        (InputTooLargeError("input too large"), 2),
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


def test_cli_displays_safe_model_validation_detail_without_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe_message = (
        "MatchAnalysis output validation failed: "
        "assessments.2.status [literal_error]: Input should be an approved status"
    )
    sensitive_detail = "TOP-SECRET-TEST-VALUE invalid-test-key"
    error = ModelOutputError(safe_message)
    error.__cause__ = ValueError(sensitive_detail)

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

    assert result.exit_code == 5
    assert safe_message in result.output
    assert sensitive_detail not in result.output
    assert "ValueError" not in result.output
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
