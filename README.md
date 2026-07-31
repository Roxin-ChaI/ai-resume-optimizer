[English](README.md) | [简体中文](README.zh-CN.md)

# AI Resume Optimizer

## Overview

AI Resume Optimizer is a Python 3.12 command-line tool for evidence-grounded
resume optimization. It accepts a text-layer PDF or DOCX resume and a target job
description supplied from a UTF-8 TXT file or pasted interactively. It uses
structured output from the OpenAI Responses API to analyze the inputs and
produces an analysis report, an optimized Markdown resume, and an editable DOCX
resume.

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

## Requirements

- Python 3.12 or later.
- An OpenAI API key.
- An OpenAI model configured to support the structured-output call used by the
  Responses API client.
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
export OPENAI_API_KEY="replace-with-your-api-key"
export OPENAI_MODEL="replace-with-a-compatible-model"
export OPENAI_TIMEOUT_SECONDS="60"
```

- `OPENAI_API_KEY` and `OPENAI_MODEL` are required.
- `OPENAI_TIMEOUT_SECONDS` is optional and defaults to `60`; it must be a positive
  finite number.

The application does not automatically load `.env` files. The
[`.env.example`](.env.example) file is a reference template only. Never commit an
API key to Git.

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

The resume and job description are sent to the configured OpenAI API for four
structured tasks: resume structuring, job-requirement extraction, match analysis,
and resume optimization. Client calls set `store=False`, but that setting must not
be interpreted as an absolute guarantee covering every provider-side retention
behavior. Review the model provider's current data policies.

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
```

Tests inject fake model clients, do not call the real OpenAI API, and do not
require a real API key.

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
            model_client.py
            models.py
            pipeline.py
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

## Release Status

The project version is `0.1.0`. No Git tag or GitHub Release is created during
this documentation and validation stage. Release operations will be handled
separately after the documentation and release checks are approved.
