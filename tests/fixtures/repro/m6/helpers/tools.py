"""A tools module with a helper base class, a mixin and private helpers; only the registered classes are tools."""
from pathlib import Path

from crewai.tools import BaseTool


def _clean(text: str) -> str:
    return text.strip()


def format_row(row: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in row.items())


class _RunScoped:
    run_id: str = ""


class _StrictTool(BaseTool, _RunScoped):
    """Base for the project's tools."""

    def _check(self, x: str) -> str:
        return _clean(x)


class SaveFileTool(_StrictTool):
    name: str = "save_file"
    description: str = "Save text to a file."

    def _run(self, text: str) -> str:
        Path("out.txt").write_text(self._check(text))
        return "saved"


class SearchTool(_StrictTool):
    name: str = "search_notes"
    description: str = "Search the notes."

    def _run(self, q: str) -> str:
        return format_row({"q": q})
