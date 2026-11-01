"""A two-agent research team (crewai): a researcher with a web search and tools loaded at runtime, a writer that
saves the report."""

import json
import os


def load_tool_specs():
    """Extra tools named in a config file: built at runtime, their names are not in the code."""
    with open(os.getenv("EXTRA_TOOLS", "extra_tools.json"), encoding="utf-8") as fh:
        return json.load(fh)


from crewai import Agent, Crew, Task
from crewai.tools import BaseTool, tool


class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Search the web."

    def _run(self, query: str) -> str:
        return query[:200]


@tool("Save Report")
def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text[:20000])
    return 'saved'


def build_tool(spec):
    return tool(spec["name"])(spec["fn"])


EXTRA = [build_tool(s) for s in load_tool_specs()]
researcher = Agent(role="Researcher", goal="find facts", backstory="You research topics for the team.",
                   tools=[WebSearchTool(), *EXTRA], llm="gpt-4o-2024-08-06", max_iter=10)
writer = Agent(role="Writer", goal="write the report", backstory="You write the final report.",
               tools=[save_report], llm="gpt-4o-2024-08-06", max_iter=10)
crew = Crew(agents=[researcher, writer],
            tasks=[Task(description="research", expected_output="notes", agent=researcher),
                   Task(description="write", expected_output="report", agent=writer)])


def main(topic: str = "x") -> str:
    return str(crew.kickoff(inputs={"topic": topic}))
