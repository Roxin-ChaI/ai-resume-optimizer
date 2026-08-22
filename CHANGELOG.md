# Changelog

## [Unreleased]

### Added

- A stable public Python API with `ResumeOptimizerConfig`,
  `ResumeOptimizerRunner`, `create_resume_optimizer`, public DTOs, and categorized
  exceptions.
- A file-free in-memory optimization path returning `OptimizationResult` with
  `output_paths == {}` for embedding in other Python applications.
- PDF and DOCX resume input, plus TXT and interactive job-description input.
- DeepSeek structured analysis and qualitative resume-to-job matching.
- Migration to DeepSeek Chat Completions JSON Output with the fixed
  `deepseek-v4-flash` model.
- Markdown analysis, Markdown resume, and editable DOCX resume output.
- A CLI with atomic-style three-file batch writing and cleanup on failure.
- English and Chinese documentation, fictitious example inputs, and CI checks.

### Safety

- Source-block evidence links and blocking of unsupported requirement claims.
- Required source-block ID manifests and exact input/output coverage constraints for
  DeepSeek resume structuring.
- Deterministic preservation of model-omitted, valid source blocks as unchanged
  `unclassified_content`, without an additional model call.
- Continued rejection of unknown source-block IDs before deterministic recovery.
- Deterministic derivation of each `ResumeSection.source_block_ids` aggregate from
  its item-level evidence IDs.
- Incorrect or missing DeepSeek section-level evidence aggregates no longer fail
  an otherwise valid resume optimization response.
- Continued preservation of item-level source-block IDs for subsequent strict
  evidence validation.
- Required requirement ID manifests with exact assessment count, uniqueness, and
  input-order constraints for DeepSeek match analysis.
- Safe model JSON and Pydantic failure diagnostics with response-model names and
  field paths, without retaining raw model responses.
- Deterministic checks for new numbers, new dates, and misplaced placeholders.
- Human-review markers for significant rewrites.
- Qualitative ratings only, with no ATS percentage score.

### Testing

- A controlled Real Public Runner E2E using the fictitious DOCX fixture and
  `deepseek-v4-flash`, validating the structured result, empty output paths,
  file-free execution, and owned-client close behavior.
- Fully offline automated tests that do not call the real DeepSeek API.
- GitHub Actions quality checks on Python 3.12.
