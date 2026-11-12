"""LangChain v1 create_agent with HumanInTheLoopMiddleware on save_record."""
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


@tool
def save_record(text: str) -> str:
    """Save a record to disk."""
    Path("record.txt").write_text(text)
    return "saved"


agent = create_agent(
    model="openai:gpt-4o-2024-08-06",
    tools=[save_record],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"save_record": True})],
    checkpointer=InMemorySaver(),
)
