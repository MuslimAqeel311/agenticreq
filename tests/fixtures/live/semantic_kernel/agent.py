"""Semantic Kernel ChatCompletionAgent with two tools."""
import asyncio
from pathlib import Path

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function


class Notes:
    @kernel_function(description="Look something up.")
    def lookup(self, q: str) -> str:
        return "hit"

    @kernel_function(description="Save a note.")
    def save_note(self, text: str) -> str:
        Path("notes.txt").write_text(text)
        return "saved"


def main(task: str = "go") -> str:
    service = OpenAIChatCompletion(ai_model_id="gpt-4o-2024-08-06", api_key="x")
    agent = ChatCompletionAgent(service=service, name="a", instructions="Help.", plugins=[Notes()])
    return str(asyncio.run(agent.get_response(messages=task)))
