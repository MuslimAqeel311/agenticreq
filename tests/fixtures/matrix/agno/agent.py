"""A two-agent research team (agno): a researcher with a web search and tools loaded at runtime, a writer that
saves the report."""

import json
import os


def load_tool_specs():
    """Extra tools named in a config file: built at runtime, their names are not in the code."""
    with open(os.getenv("EXTRA_TOOLS", "extra_tools.json"), encoding="utf-8") as fh:
        return json.load(fh)


from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team
from agno.tools import tool


@tool(name="web-search")
def web_search(query: str) -> str:
    """Search the web."""
    return query[:200]


@tool(name="Save Report")
def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text[:20000])
    return 'saved'


EXTRA = [tool(name=s["name"])(s["fn"]) for s in load_tool_specs()]
researcher = Agent(name="researcher", model=OpenAIChat(id="gpt-4o-2024-08-06"), instructions="You research topics.",
                   tools=[web_search, *EXTRA], tool_call_limit=10)
writer = Agent(name="writer", model=OpenAIChat(id="gpt-4o-2024-08-06"), instructions="You write the final report.",
               tools=[save_report], tool_call_limit=10)
team = Team(name="team", members=[researcher, writer], model=OpenAIChat(id="gpt-4o-2024-08-06"))


def main(topic: str = "x") -> str:
    return str(team.run(topic).content)
