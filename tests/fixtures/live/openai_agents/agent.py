"""OpenAI Agents SDK with two tools."""
from pathlib import Path

from agents import Agent, Runner, function_tool


@function_tool
def lookup(q: str) -> str:
    """Look something up."""
    return "hit"


@function_tool
def save_note(text: str) -> str:
    """Save a note."""
    Path("notes.txt").write_text(text)
    return "saved"


def main(task: str = "go") -> str:
    agent = Agent(name="a", instructions="Help.", tools=[lookup, save_note], model="gpt-4o-2024-08-06")
    return str(Runner.run_sync(agent, task, max_turns=6).final_output)
