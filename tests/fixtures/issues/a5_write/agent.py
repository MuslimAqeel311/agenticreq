"""The tool writes outside allow_write: and does not handle an error."""

from pathlib import Path

from minifw import Agent, tool


@tool
def save_report(text: str) -> str:
    """Save the report to disk."""
    Path("out").mkdir(exist_ok=True)
    Path("out/report.txt").write_text(text)
    return Path("out/report.txt").read_text()


def main(task: str = "write") -> str:
    return Agent(name="writer", tools=[save_report]).invoke(task, max_turns=4)
