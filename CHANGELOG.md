# Changelog

## [Unreleased]

### Added

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
- Required requirement ID manifests with exact assessment count, uniqueness, and
  input-order constraints for DeepSeek match analysis.
- Safe model JSON and Pydantic failure diagnostics with response-model names and
  field paths, without retaining raw model responses.
- Deterministic checks for new numbers, new dates, and misplaced placeholders.
- Human-review markers for significant rewrites.
- Qualitative ratings only, with no ATS percentage score.

### Testing

- Fully offline automated tests that do not call the real DeepSeek API.
- GitHub Actions quality checks on Python 3.12.
