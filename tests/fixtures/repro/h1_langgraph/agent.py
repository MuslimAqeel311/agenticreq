"""LangGraph: interrupt() inside the tool."""
from pathlib import Path

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt


@tool
def save_record(text: str) -> str:
    """Save a record to disk."""
    if interrupt({"question": f"save {text}?"}) != "yes":
        return "not approved"
    Path("record.txt").write_text(text)
    return "saved"


agent = create_react_agent("openai:gpt-4o-2024-08-06", tools=[save_record])
