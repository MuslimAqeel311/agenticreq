"""Haystack Agent with a confirmation strategy on save_record."""
from pathlib import Path

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.human_in_the_loop import AlwaysAskPolicy, BlockingConfirmationStrategy, SimpleConsoleUI
from haystack.tools import tool


@tool
def save_record(text: str) -> str:
    """Save a record to disk."""
    Path("record.txt").write_text(text)
    return "saved"


agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-2024-08-06"),
    tools=[save_record],
    confirmation_strategies={"save_record": BlockingConfirmationStrategy(AlwaysAskPolicy(), SimpleConsoleUI())},
)
