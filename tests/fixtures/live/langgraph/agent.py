"""LangGraph ReAct agent with two tools."""
from pathlib import Path

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return "hit"


@tool
def save_note(text: str) -> str:
    """Save a note."""
    Path("notes.txt").write_text(text)
    return "saved"


def main(task: str = "go") -> str:
    agent = create_react_agent(ChatOpenAI(model="gpt-4o-2024-08-06"), tools=[lookup, save_note])
    return str(agent.invoke({"messages": [("user", task)]}, {"recursion_limit": 8}))
