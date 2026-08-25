[English](README.md) | [简体中文](README.zh-CN.md)

# AI Resume Optimizer

## Overview

AI Resume Optimizer is a Python 3.12 library and command-line tool for
evidence-grounded resume optimization. Its public Python API accepts a text-layer
PDF or DOCX path plus job-description text and returns a validated, in-memory
result. The compatible CLI also accepts a UTF-8 TXT job description or interactive
input and exports an analysis report, an optimized Markdown resume, and an editable
DOCX resume.

The tool is designed to rewrite conservatively. It must not invent experience,
skills, employers, education, dates, metrics, or other facts merely to improve a
match.

## Features

- Extracts text from text-layer PDFs and from DOCX headings, paragraphs, lists,
  and tables.
- Structures job requirements and resume content with validated Pydantic models.
- Reports only qualitative match ratings: `高`, `一般`, or `低`.
- Links factual resume items and assessments to source block IDs.
- Blocks optimized content containing unsupported new numbers or dates.
- Blocks unsupported job requirements from being presented as established facts.
- Requires significant rewrites to be marked for human review.
- Generates Markdown analysis, Markdown resume, and editable DOCX resume files.
- Provides a stable in-memory Python runner for embedding the optimizer in other
  applications without writing files.
- Refuses to overwrite any existing output file by default.
- Includes repeatable offline tests using fake model clients.

It does not produce percentages, ATS scores, pass rates, or recruiting-platform
simulations.

## How It Works

One optimization run follows this order:

1. Preflight the three output paths and reject existing output files.
2. Normalize and validate the job description.
3. Parse the PDF or DOCX resume into ordered source blocks.
4. Structure the resume while retaining source-block evidence.
5. Extract a structured job profile and requirements.
6. Analyze each requirement against resume evidence.
7. Produce an evidence-linked optimized resume.
8. Run deterministic truthfulness checks.
9. Render the analysis report and resume as Markdown and DOCX.
10. Write the three output files as one batch, cleaning up partial output on failure.

All model responses are validated as explicit Pydantic data models before the
pipeline uses them.

The public runner performs the in-memory stages without output-path preflight,
rendering, or file writing. The CLI adds those export responsibilities around the
same shared optimization core.

## Requirements

- Python 3.12 or later.
- A DeepSeek API key.
- The supported DeepSeek model, `deepseek-v4-flash`.
- A PDF with an extractable text layer, or a readable DOCX file.

Scanned PDFs without a text layer are not supported.

## Installation

Create a virtual environment and install from the repository source:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

The `dev` extra installs pytest and Ruff. For ordinary use from a source checkout,
install the project without the extra:

```sh
.venv/bin/python -m pip install -e .
```

This repository does not claim a published package distribution.

## Configuration

Set these environment variables through your shell, execution environment, or
secret manager:

```sh
export DEEPSEEK_API_KEY="replace-with-your-deepseek-api-key"
export DEEPSEEK_MODEL="deepseek-v4-flash"
export DEEPSEEK_TIMEOUT_SECONDS="60"
```

- `DEEPSEEK_API_KEY` is required.
- `DEEPSEEK_MODEL` is optional and defaults to `deepseek-v4-flash`; this is the
  only supported model value.
- `DEEPSEEK_TIMEOUT_SECONDS` is optional and defaults to `60`; it must be a
  positive finite number.
- The DeepSeek API base URL is fixed at `https://api.deepseek.com` and is not
  configurable.

The project uses the OpenAI Python SDK only as a client for DeepSeek's
OpenAI-compatible API. It does not support OpenAI models or the Responses API.

The application does not automatically load `.env` files. The
[`.env.example`](.env.example) file is a reference template only. Never commit an
API key to Git.

## Public Python API

Version 0.2.1 provides a stable public integration boundary:

```python
from pathlib import Path

from ai_resume_optimizer import (
    ResumeOptimizerConfig,
    create_resume_optimizer,
)

config = ResumeOptimizerConfig(
    deepseek_api_key="replace-with-your-deepseek-api-key",
)
runner = create_resume_optimizer(config)

try:
    result = runner.optimize(
        resume_path=Path("resume.docx"),
        job_description="A concise job description.",
    )
finally:
    runner.close()
```

The public contract includes:

- `ResumeOptimizerConfig`: immutable production configuration. Its representation
  excludes `deepseek_api_key`.
- `create_resume_optimizer(config)`: composes the supported DeepSeek client and an
  owned `ResumeOptimizerRunner`.
- `ResumeOptimizerRunner`: accepts an injected `ModelClient` and exposes
  `optimize(...)` and idempotent `close()`.
- `ModelClient`: the provider-neutral structured-generation dependency-injection
  protocol.
- `OptimizationResult`: the validated result returned by the runner.
- Public DTOs for analysis and optimized-resume content, plus categorized public
  exceptions rooted at `ResumeOptimizerError`.

`ResumeOptimizerRunner.optimize` accepts only:

- `resume_path: Path`: an existing text-layer PDF or readable DOCX resume. Resume
  text is limited to 50,000 normalized characters.
- `job_description: str`: non-empty plain text limited to 30,000 normalized
  characters.

The result contains `analysis`, `optimized_resume`, `warnings`, and `output_paths`.
`analysis` exposes `overall_rating`, `overall_evaluation`, requirement
`assessments`, `main_issues`, `section_suggestions`, `keyword_suggestions`,
`truthfulness_risks`, and `content_not_to_add`. `optimized_resume` exposes validated
sections, pending user inputs, and warnings. The API does not invent an ATS score,
confidence value, token usage, or metrics.

### Public provenance API

Each production `RequirementAssessment` retains its existing `requirement_id`,
`source_block_ids`, status, reason, and suggested action, and now also exposes:

- `requirement: RequirementReference`: the human-readable requirement description,
  category, importance, and exact job-description source excerpt.
- `evidence: list[RequirementEvidence]`: the original source-block kind, location,
  excerpt, and any explicit semantic section references.

Requirement provenance is joined by stable requirement IDs. Evidence is copied
deterministically from the original parsed `SourceBlock` objects in the exact order
of `source_block_ids`; it is not regenerated by the model, inferred from the
optimized resume, or matched fuzzily. Unknown requirement or source-block IDs fail
closed with `ModelOutputError`.

This is an additive v0.2.1 contract change. The Runner API and CLI are unchanged,
and the existing `requirement_id` and `source_block_ids` fields remain compatible.
Legacy manual DTO construction may omit the new fields; normal production Runner
results populate the requirement reference and aligned evidence list.

For the public runner, `output_paths == {}`. It does not write Markdown or DOCX,
create an output directory, or print to standard output. File export remains a
separate CLI capability.

The runner raises stable domain exceptions such as `InputError`,
`ResumeExtractionError`, `ModelCallError`, `ModelOutputError`, `TruthfulnessError`,
and `ResumeOptimizerClosedError`. Embedded callers can catch these exceptions
without parsing CLI exit text, standard error, or provider-SDK exceptions.

The public runner does not accept resume TXT, bytes, upload objects, URLs, scanned
PDFs, or OCR input. The CLI can read the job description from a UTF-8 TXT file, but
the public runner receives job-description text directly as `str`.

## Architecture and Lifecycle

```text
Application / CLI
        ↓
ResumeOptimizerRunner
        ↓
in-memory optimization core
        ↓
ModelClient
        ↓
DeepSeekModelClient
```

Production composition is
`ResumeOptimizerConfig → DeepSeekModelClient → owned ResumeOptimizerRunner`.
Factory-created runners own and close their provider client. A model client passed
directly to `ResumeOptimizerRunner` is external by default and is not closed by the
runner. `close()` is idempotent, and optimization after close raises
`ResumeOptimizerClosedError`.

The in-memory optimization core is separate from Markdown/DOCX rendering and the
shared atomic file-export step.

## Usage

Optimize a PDF resume with a TXT job description:

```sh
ai-resume-optimizer optimize \
  --resume ./resume.pdf \
  --job-description ./job_description.txt \
  --output-dir ./output
```

Use a DOCX resume:

```sh
ai-resume-optimizer optimize \
  --resume ./resume.docx \
  --job-description ./job_description.txt
```

Paste the job description interactively by omitting `--job-description`:

```sh
ai-resume-optimizer optimize \
  --resume ./resume.docx
```

Enter `END` on a line by itself, or send EOF, to finish interactive input. Piped
non-interactive standard input is not supported when no job-description file is
provided.

There is no `--overwrite` option. If any expected output file already exists,
the command refuses to run.

The v0.2.1 CLI keeps its existing arguments, environment variables, three output
files, overwrite protection, and categorized exit codes. Internally its production
path is now:

```text
environment
→ ResumeOptimizerConfig
→ create_resume_optimizer
→ ResumeOptimizerRunner.optimize (exactly once)
→ shared atomic export
```

## Output Files

The output directory contains:

- `analysis_report.md`
- `optimized_resume.md`
- `optimized_resume.docx`

The three files come from the same run. Both resume formats are rendered directly
from the same validated `OptimizedResume` model; the DOCX is not produced by
re-parsing the Markdown. The analysis report is not an ATS score or a prediction
of hiring outcomes. Review every output before use.

## Truthfulness and Review

Deterministic checks block:

- Unknown source block IDs.
- Unknown requirement IDs.
- Unsupported requirements linked to factual resume content.
- Numbers not present in the cited source blocks.
- Dates not present in the cited source blocks.
- Obvious placeholder text in factual resume content.
- Significant rewrites that are not marked for human review.

These checks are conservative safeguards, not complete semantic verification.
They cannot reliably identify every newly introduced company, school, skill, or
certificate; every semantic exaggeration; every escalation of responsibility;
or every non-equivalent paraphrase. Prompt instructions also are not a factual
guarantee. The user must review the final resume against the original evidence.

## Privacy

The resume and job description are sent to the DeepSeek API for four structured
tasks: resume structuring, job-requirement extraction, match analysis, and resume
optimization. The project does not claim that DeepSeek stores or never stores
requests. Review DeepSeek's current data policies before sending sensitive data.

Generated files are written to the local output directory selected by the user.
The tool has no database, cloud file store, or optimization-history service. Do
not expose API keys or sensitive resume content in public terminals, logs, or
repositories.

## Examples

The example inputs are entirely fictitious:

- [Example guide](examples/README.md)
- [Sample DOCX resume](examples/sample_resume.docx)
- [Sample job description](examples/sample_job_description.txt)

Pre-generated model outputs are intentionally not included.

## Testing

Run the offline test and quality checks:

```sh
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m pip check
```

Tests inject fake model clients, do not call the real DeepSeek API, and do not
require a real API key.

The v0.2.1 release baseline is 336 passing offline tests, Ruff passing,
`ruff format --check` passing, and `pip check` passing. This project does not
currently configure mypy.

A controlled Real Public Runner E2E also passed with the fictitious
`examples/sample_resume.docx` fixture and `deepseek-v4-flash`: production factory
composition passed, the result validated as `OptimizationResult`,
`output_paths == {}`, no output files were created, and runner close succeeded.

An initial run received an empty model response and was rejected by
`ModelOutputError`. A subsequent safe metadata diagnostic returned a normal
`ChatCompletion` response, and the final controlled full E2E passed. This does not
establish a client bug or provider bug, and v0.2.1 does not add automatic retry,
backoff, or fallback behavior.

A controlled Real Public Runner Provenance E2E also passed using the same fictitious
fixture and supported model. It verified non-empty human-readable requirement
references, original evidence excerpts, exact evidence/source-ID ordering, the
production provenance invariant, Pydantic round-trip validation,
`output_paths == {}`, no file side effects, and idempotent close. No resume, job
description, prompt, raw model response, or API key is stored in this repository.

## Project Structure

```text
ai-resume-optimizer/
    src/
        ai_resume_optimizer/
            parsers/
            prompts/
            renderers/
            services/
            cli.py
            config.py
            factory.py
            model_client.py
            models.py
            pipeline.py
            runner.py
    tests/
        fakes/
        integration/
        unit/
    examples/
    .github/
        workflows/
    pyproject.toml
    README.md
    README.zh-CN.md
```

## Limitations

- No OCR or scanned-PDF recognition.
- No restoration of complex multi-column PDF or DOCX layouts.
- No `--overwrite` option.
- No PDF resume export.
- No job-page scraping or recruiting-platform login.
- No automatic job applications.
- No ATS score or pass-rate estimate; the tool does not guarantee an interview
  or hiring outcome.
- Truthfulness checks are conservative rules, not complete semantic verification.
- Generated DOCX files use simple built-in document structures and do not restore
  the source template.
- Oversized inputs are rejected rather than silently truncated.
- The public runner does not accept resume TXT, bytes, upload objects, or URLs.
- Invalid or empty model output is rejected; automatic retry and fallback models
  are not implemented.

## Release Status

The project version is `0.2.1`. No v0.2.1 Git tag or GitHub Release is created during
this documentation and validation stage. Release operations will be handled
separately after the documentation and release checks are approved.
