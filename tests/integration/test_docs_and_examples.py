"""Offline checks for release documentation, examples, and CI configuration."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from ai_resume_optimizer.config import load_settings
from ai_resume_optimizer.exceptions import ConfigurationError
from ai_resume_optimizer.parsers import parse_resume
from ai_resume_optimizer.parsers.job_description import read_job_description

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LANGUAGE_LINKS = "[English](README.md) | [简体中文](README.zh-CN.md)"
SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
ABSOLUTE_USER_PATH_PATTERN = re.compile(
    r"(?:/" + "Users" + r"/|/" + "home" + r"/|[A-Za-z]:[\\/](?:Users|Documents)[\\/])"
)


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_readmes_are_bilingual_accurate_and_safe() -> None:
    english = _read("README.md")
    chinese = _read("README.zh-CN.md")

    assert english.strip()
    assert chinese.strip()
    assert english.startswith(LANGUAGE_LINKS)
    assert chinese.startswith(LANGUAGE_LINKS)

    for expected in (
        "ai-resume-optimizer optimize",
        "DEEPSEEK_API_KEY",
        "deepseek-v4-flash",
        "analysis_report.md",
        "optimized_resume.md",
        "optimized_resume.docx",
        "No OCR",
        "not an ATS score",
        "does not guarantee",
    ):
        assert expected.casefold() in english.casefold()

    for expected in (
        "ai-resume-optimizer optimize",
        "DEEPSEEK_API_KEY",
        "deepseek-v4-flash",
        "analysis_report.md",
        "optimized_resume.md",
        "optimized_resume.docx",
        "不支持 OCR",
        "不是 ATS 分数",
        "不保证面试或录用",
    ):
        assert expected.casefold() in chinese.casefold()

    combined = english + chinese
    assert not ABSOLUTE_USER_PATH_PATTERN.search(combined)
    assert not SECRET_PATTERN.search(combined)
    assert "OpenAI Responses API" not in combined
    assert "OpenAI API key" not in combined
    assert "configured OpenAI API" not in combined
    assert "github release has been created" not in english.casefold()
    assert "已创建 github release" not in chinese.casefold()
    assert "pip install ai-resume-optimizer" not in combined


def test_environment_example_contains_only_approved_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (PROJECT_ROOT / ".env.example").is_file()
    env_example = _read(".env.example")
    assignments = [
        line.strip()
        for line in env_example.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    parsed = dict(line.split("=", maxsplit=1) for line in assignments)

    assert set(parsed) == {
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_TIMEOUT_SECONDS",
    }
    assert parsed["DEEPSEEK_MODEL"] == "deepseek-v4-flash"
    assert float(parsed["DEEPSEEK_TIMEOUT_SECONDS"]) > 0
    assert "OPENAI_" not in env_example
    assert not SECRET_PATTERN.search(env_example)

    for name in parsed:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY"):
        load_settings()


def test_gitignore_excludes_local_environment_file() -> None:
    assert (PROJECT_ROOT / ".gitignore").is_file()

    ignore_lines = {
        line.strip()
        for line in _read(".gitignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert ".env" in ignore_lines
    assert ".env.example" not in ignore_lines


def test_changelog_records_only_unreleased_work() -> None:
    changelog = _read("CHANGELOG.md")

    assert "## [Unreleased]" in changelog
    assert "### Added" in changelog
    assert "### Safety" in changelog
    assert "### Testing" in changelog
    assert not re.search(r"## \[0\.1\.0\]\s*-\s*\d{4}-\d{2}-\d{2}", changelog)
    assert "tag has been created" not in changelog.casefold()
    assert "release has been created" not in changelog.casefold()


def test_sample_job_description_is_readable_and_fictitious() -> None:
    path = PROJECT_ROOT / "examples/sample_job_description.txt"
    description = read_job_description(path)

    assert description
    assert "required qualifications" in description.casefold()
    assert "preferred qualifications" in description.casefold()
    assert not re.search(r"https?://", description)
    assert not SECRET_PATTERN.search(description)


def test_sample_resume_is_parseable_and_unchanged() -> None:
    path = PROJECT_ROOT / "examples/sample_resume.docx"
    before_hash = _sha256(path)
    before_mtime = path.stat().st_mtime_ns

    resume = parse_resume(path)

    assert resume.source_format == "docx"
    assert resume.blocks
    assert resume.plain_text
    for section_name in ("Summary", "Skills", "Experience", "Projects", "Education"):
        assert section_name in resume.plain_text
    assert _sha256(path) == before_hash
    assert path.stat().st_mtime_ns == before_mtime
    assert not ABSOLUTE_USER_PATH_PATTERN.search(resume.plain_text)
    assert not SECRET_PATTERN.search(resume.plain_text)


def test_example_guide_describes_safe_usage() -> None:
    guide = _read("examples/README.md")

    assert "sample_resume.docx" in guide
    assert "sample_job_description.txt" in guide
    assert "ai-resume-optimizer optimize" in guide
    assert "DEEPSEEK_API_KEY" in guide
    assert "deepseek-v4-flash" in guide
    assert "fictitious" in guide.casefold()
    assert "does not include generated example outputs" in guide
    assert "refuses to overwrite" in guide


def test_ci_runs_offline_quality_checks_without_release_steps() -> None:
    ci = _read(".github/workflows/ci.yml")
    ci_lower = ci.casefold()

    for expected in (
        "actions/checkout@v7",
        "actions/setup-python@v7",
        'python-version: "3.12"',
        "permissions:",
        "contents: read",
        "python -m pytest",
        "python -m ruff check .",
        "python -m ruff format --check .",
        "ai-resume-optimizer --help",
        "ai-resume-optimizer optimize --help",
    ):
        assert expected.casefold() in ci_lower

    assert "deepseek_api_key" not in ci_lower
    assert "openai_api_key" not in ci_lower
    assert "secrets." not in ci_lower
    assert "git push" not in ci_lower
    assert "publish" not in ci_lower
    assert "release" not in ci_lower
