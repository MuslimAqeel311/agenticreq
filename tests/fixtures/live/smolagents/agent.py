"""smolagents ToolCallingAgent with two tools."""
from pathlib import Path

from smolagents import OpenAIServerModel, ToolCallingAgent, tool


@tool
def lookup(q: str) -> str:
    """Look something up.

    Args:
        q: the query
    """
    return "hit"


@tool
def save_note(text: str) -> str:
    """Save a note.

    Args:
        text: the note
    """
    Path("notes.txt").write_text(text)
    return "saved"


def main(task: str = "go") -> str:
    model = OpenAIServerModel(model_id="gpt-4o-2024-08-06", api_key="x")
    return str(ToolCallingAgent(tools=[lookup, save_note], model=model, max_steps=4).run(task))
