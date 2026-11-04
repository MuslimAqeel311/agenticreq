"""PydanticAI with two tools."""
from pathlib import Path

from pydantic_ai import Agent


def lookup(q: str) -> str:
    """Look something up."""
    return "hit"


def save_note(text: str) -> str:
    """Save a note."""
    Path("notes.txt").write_text(text)
    return "saved"


def main(task: str = "go") -> str:
    agent = Agent("openai:gpt-4o-2024-08-06", tools=[lookup, save_note])
    return str(agent.run_sync(task).output)
