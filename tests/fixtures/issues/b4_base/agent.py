"""A helper base class under the real tools. Only `SaveFile` is passed to an agent."""

from crewai import Agent
from crewai.tools import BaseTool


class _RunScoped(BaseTool):
    """Shared state for this project's tools."""

    def _scoped(self) -> str:
        return "run"

    def _run(self, q: str) -> str:
        raise NotImplementedError


class SaveFile(_RunScoped):
    name: str = "save_file"
    description: str = "Save a file."

    def _run(self, q: str) -> str:
        return q


class Unused(_RunScoped):
    name: str = "unused_tool"
    description: str = "Never registered."

    def _run(self, q: str) -> str:
        return q


agent = Agent(role="writer", goal="write", backstory="b", tools=[SaveFile()], llm="gpt-4o-2024-08-06")
