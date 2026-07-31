# Changelog

## [Unreleased]

### Added

- PDF and DOCX resume input, plus TXT and interactive job-description input.
- OpenAI structured analysis and qualitative resume-to-job matching.
- Markdown analysis, Markdown resume, and editable DOCX resume output.
- A CLI with atomic-style three-file batch writing and cleanup on failure.
- English and Chinese documentation, fictitious example inputs, and CI checks.

### Safety

- Source-block evidence links and blocking of unsupported requirement claims.
- Deterministic checks for new numbers, new dates, and misplaced placeholders.
- Human-review markers for significant rewrites.
- Qualitative ratings only, with no ATS percentage score.

### Testing

- Fully offline automated tests that do not call the real OpenAI API.
- GitHub Actions quality checks on Python 3.12.
