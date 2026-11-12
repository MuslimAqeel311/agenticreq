"""Plain input() with no time limit; the spec asks for a 1h timeout."""
from pathlib import Path

from agents import Agent, function_tool


@function_tool
def save_record(text: str) -> str:
    """Save a record to disk."""
    if input("approve? ") != "y":
        return "not approved"
    Path("record.txt").write_text(text)
    return "saved"


agent = Agent(name="saver", model="gpt-4o-2024-08-06", tools=[save_record])
