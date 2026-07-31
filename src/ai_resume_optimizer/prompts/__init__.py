"""Load the fixed prompt resources approved for stage 3."""

from importlib.resources import files

_APPROVED_PROMPTS = frozenset({"structure_resume.txt", "analyze_job.txt"})


def load_prompt(name: str) -> str:
    """Load one approved UTF-8 prompt without depending on the working directory."""

    if name not in _APPROVED_PROMPTS:
        raise ValueError(f"Unknown prompt resource: {name!r}.")

    content = files(__package__).joinpath(name).read_text(encoding="utf-8")
    if not content.strip():
        raise RuntimeError(f"Prompt resource is empty: {name!r}.")
    return content


__all__ = ["load_prompt"]
