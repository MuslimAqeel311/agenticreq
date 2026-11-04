"""Agno agent with two tools."""
from pathlib import Path

from agno.agent import Agent
from agno.models.openai import OpenAIChat


def lookup(q: str) -> str:
    """Look something up."""
    return "hit"


def save_note(text: str) -> str:
    """Save a note."""
    Path("notes.txt").write_text(text)
    return "saved"


def main(task: str = "go") -> str:
    agent = Agent(model=OpenAIChat(id="gpt-4o-2024-08-06"), tools=[lookup, save_note], tool_call_limit=4)
    return str(agent.run(task).content)
