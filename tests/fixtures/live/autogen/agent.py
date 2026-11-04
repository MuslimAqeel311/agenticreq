"""AutoGen AgentChat with two tools."""
import asyncio
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


def lookup(q: str) -> str:
    """Look something up."""
    return "hit"


def save_note(text: str) -> str:
    """Save a note."""
    Path("notes.txt").write_text(text)
    return "saved"


def main(task: str = "go") -> str:
    client = OpenAIChatCompletionClient(model="gpt-4o-2024-08-06")
    agent = AssistantAgent("a", model_client=client, tools=[lookup, save_note], max_tool_iterations=4)
    return str(asyncio.run(agent.run(task=task)))
