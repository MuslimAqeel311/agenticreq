"""LlamaIndex FunctionAgent with two tools."""
import asyncio
from pathlib import Path

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI


def lookup(q: str) -> str:
    """Look something up."""
    return "hit"


def save_note(text: str) -> str:
    """Save a note."""
    Path("notes.txt").write_text(text)
    return "saved"


async def go(task: str) -> str:
    agent = FunctionAgent(name="a", tools=[lookup, save_note], llm=OpenAI(model="gpt-4o-2024-08-06"), system_prompt="Help.")
    return str(await agent.run(task))


def main(task: str = "go") -> str:
    return asyncio.run(go(task))
