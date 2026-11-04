"""CrewAI crew with two tools."""
from pathlib import Path

from crewai import Agent, Crew, Task
from crewai.tools import tool


@tool("lookup")
def lookup(q: str) -> str:
    """Look something up."""
    return "hit"


@tool("save_note")
def save_note(text: str) -> str:
    """Save a note."""
    Path("notes.txt").write_text(text)
    return "saved"


def main(task: str = "go") -> str:
    agent = Agent(role="helper", goal="help", backstory="b", tools=[lookup, save_note], llm="gpt-4o-2024-08-06", max_iter=4)
    t = Task(description=task, expected_output="an answer", agent=agent)
    return str(Crew(agents=[agent], tasks=[t]).kickoff())
