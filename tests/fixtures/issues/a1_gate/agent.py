"""A write tool the spec says needs approval. The code has no gate, and mocks.yml patches nothing."""

from pathlib import Path

from minifw import Agent, tool


@tool
def save_report(text: str) -> str:
    """Save the report to disk."""
    Path("report.txt").write_text(text)
    return "saved"


@tool
def search(q: str) -> str:
    """Search the notes."""
    return "nothing"


def main(task: str = "write the report") -> str:
    return Agent(name="writer", tools=[search, save_report]).invoke(task, max_turns=6)
