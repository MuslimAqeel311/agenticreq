"""Two CrewAI agents: this framework has no setting for a whole run."""

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool


@tool("lookup")
def lookup(q: str) -> str:
    """Look up."""
    return q


a = Agent(role="a", goal="g", backstory="b", tools=[lookup], allow_delegation=True, llm="gpt-4o-2024-08-06")
b = Agent(role="b", goal="g", backstory="b", tools=[lookup], llm="gpt-4o-2024-08-06")
crew = Crew(agents=[a, b], tasks=[Task(description="d", expected_output="e", agent=a)], process=Process.sequential)
