"""One tool class, named per instance; another names itself with a class attribute of a base."""

from crewai import Agent
from crewai.tools import BaseTool


class _Tool(BaseTool):
    name: str = "tool"
    description: str = "A tool."

    def _run(self, q: str) -> str:
        return q


save = _Tool(name="save_file", description="Save a file.")
load = _Tool(name="load_file", description="Load a file.")


class Named(BaseTool):
    name = "lookup_item"
    description = "Look up an item."

    def _run(self, q: str) -> str:
        return q


agent = Agent(role="writer", goal="write", backstory="b", tools=[save, load, Named()], llm="gpt-4o-2024-08-06")
