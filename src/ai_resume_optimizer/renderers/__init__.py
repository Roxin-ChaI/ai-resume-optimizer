"""Public renderer entry points."""

from ai_resume_optimizer.renderers.docx import (
    render_optimized_resume as render_optimized_resume_docx,
)
from ai_resume_optimizer.renderers.markdown import (
    render_analysis_report as render_analysis_report_markdown,
)
from ai_resume_optimizer.renderers.markdown import (
    render_optimized_resume as render_optimized_resume_markdown,
)

__all__ = [
    "render_analysis_report_markdown",
    "render_optimized_resume_docx",
    "render_optimized_resume_markdown",
]
