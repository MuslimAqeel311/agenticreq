"""OpenAI Agents: save_record needs approval."""
from pathlib import Path

from agents import Agent, function_tool


@function_tool(needs_approval=True)
def save_record(text: str) -> str:
    """Save a record to disk."""
    Path("record.txt").write_text(text)
    return "saved"


agent = Agent(name="saver", model="gpt-4o-2024-08-06", tools=[save_record])
