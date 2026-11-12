"""Agno: the tool asks for confirmation."""
from pathlib import Path

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool


@tool(requires_confirmation=True)
def save_record(text: str) -> str:
    """Save a record to disk."""
    Path("record.txt").write_text(text)
    return "saved"


agent = Agent(model=OpenAIChat(id="gpt-4o-2024-08-06"), tools=[save_record])
