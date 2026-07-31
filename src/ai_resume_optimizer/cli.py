"""Typer command-line interface for one resume optimization run."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

import typer

from ai_resume_optimizer.config import load_settings
from ai_resume_optimizer.exceptions import InputError, ResumeOptimizerError
from ai_resume_optimizer.model_client import OpenAIModelClient
from ai_resume_optimizer.parsers.job_description import (
    normalize_job_description,
    read_job_description,
)
from ai_resume_optimizer.pipeline import run_optimization

app = typer.Typer(
    add_completion=False,
    help="Optimize an evidence-grounded resume for a target job.",
    no_args_is_help=True,
)


@app.callback()
def application() -> None:
    """AI Resume Optimizer command group."""


def _stdin() -> TextIO:
    return sys.stdin


def _read_interactive_job_description() -> str:
    stream = _stdin()
    if not stream.isatty():
        raise InputError(
            "A job-description file is required when standard input is not interactive."
        )

    typer.echo("Paste the job description. Enter END on its own line, or send EOF, to finish.")
    lines: list[str] = []
    while True:
        line = stream.readline()
        if line == "":
            break
        line = line.rstrip("\r\n")
        if line.strip() == "END":
            break
        lines.append(line)
    return normalize_job_description("\n".join(lines))


def _execute_optimization(
    *,
    resume: Path,
    job_description_path: Path | None,
    output_dir: Path,
) -> None:
    job_description = (
        read_job_description(job_description_path)
        if job_description_path is not None
        else _read_interactive_job_description()
    )
    settings = load_settings()
    model_client = OpenAIModelClient(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        timeout_seconds=settings.openai_timeout_seconds,
    )
    result = run_optimization(
        resume_path=resume,
        job_description=job_description,
        output_dir=output_dir,
        model_client=model_client,
    )
    typer.echo(f"Analysis report: {result.output_paths['analysis_report']}")
    typer.echo(f"Markdown resume: {result.output_paths['optimized_resume_markdown']}")
    typer.echo(f"DOCX resume: {result.output_paths['optimized_resume_docx']}")
    if result.warnings:
        typer.echo(f"Warnings: {len(result.warnings)}. Review them before use.")
    typer.echo("Review all generated content before use.")


@app.command()
def optimize(
    resume: Path = typer.Option(..., "--resume", help="PDF or DOCX resume path."),
    job_description: Path | None = typer.Option(
        None,
        "--job-description",
        help="UTF-8 TXT job-description path.",
    ),
    output_dir: Path = typer.Option(
        Path("output"),
        "--output-dir",
        help="Directory for the three generated files.",
    ),
) -> None:
    """Optimize a resume for one target job."""

    try:
        _execute_optimization(
            resume=resume,
            job_description_path=job_description,
            output_dir=output_dir,
        )
    except ResumeOptimizerError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=error.exit_code) from None
    except (OSError, RuntimeError, TypeError, ValueError):
        typer.echo("Error: An unexpected internal error occurred.", err=True)
        raise typer.Exit(code=1) from None


def main() -> None:
    """Run the Typer application."""

    app()


if __name__ == "__main__":
    main()
