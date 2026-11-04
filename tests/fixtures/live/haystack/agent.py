"""Haystack Agent with two tools."""
from pathlib import Path

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack.utils import Secret


def lookup(q: str) -> str:
    """Look something up."""
    return "hit"


def save_note(text: str) -> str:
    """Save a note."""
    Path("notes.txt").write_text(text)
    return "saved"


TOOLS = [
    Tool(name="lookup", description="Look something up.", function=lookup,
         parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
    Tool(name="save_note", description="Save a note.", function=save_note,
         parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
]


def main(task: str = "go") -> str:
    gen = OpenAIChatGenerator(model="gpt-4o-2024-08-06", api_key=Secret.from_token("x"))
    agent = Agent(chat_generator=gen, tools=TOOLS, max_agent_steps=5)
    return str(agent.run(messages=[ChatMessage.from_user(task)]))
