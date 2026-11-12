"""Google ADK: a before_tool_callback blocks save_record unless a person approved."""
from pathlib import Path

from google.adk.agents import Agent


def save_record(text: str) -> str:
    """Save a record to disk."""
    Path("record.txt").write_text(text)
    return "saved"


def check(tool, args, tool_context):
    if tool.name == "save_record" and not tool_context.state.get("approved"):
        return {"result": "a person must approve save_record first"}
    return None


root_agent = Agent(name="saver", model="gemini-2.0-flash-001", tools=[save_record], before_tool_callback=check)
