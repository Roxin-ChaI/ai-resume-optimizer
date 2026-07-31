# Fictitious Input Examples

All data in this directory is fictitious and exists only to demonstrate the
supported input formats. It is not derived from a real person's resume or a real
job posting.

The repository does not include generated example outputs because a real run
requires the user's own DeepSeek API configuration. Before running the example,
set `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, and `DEEPSEEK_TIMEOUT_SECONDS` through your
shell or execution environment. The application does not automatically load
`.env.example`.

`DEEPSEEK_MODEL` defaults to and only accepts `deepseek-v4-flash`. The API base
URL is fixed at `https://api.deepseek.com`.

Run:

```sh
ai-resume-optimizer optimize \
  --resume examples/sample_resume.docx \
  --job-description examples/sample_job_description.txt \
  --output-dir examples/output
```

A successful run creates:

- `examples/output/analysis_report.md`
- `examples/output/optimized_resume.md`
- `examples/output/optimized_resume.docx`

The tool refuses to overwrite any of these files if they already exist. Review
all generated content against the source resume before use.

Do not commit real resumes or other sensitive personal information to a public
repository.
