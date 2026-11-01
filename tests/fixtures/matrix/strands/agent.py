"""A two-agent research team (strands): a researcher with a web search and tools loaded at runtime, a writer that
saves the report."""

import json
import os


def load_tool_specs():
    """Extra tools named in a config file: built at runtime, their names are not in the code."""
    with open(os.getenv("EXTRA_TOOLS", "extra_tools.json"), encoding="utf-8") as fh:
        return json.load(fh)


from strands import Agent, tool
from strands.models.openai import OpenAIModel


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
model = OpenAIModel(model_id="gpt-4o-2024-08-06")
writer = Agent(name="writer", model=model, system_prompt="You write the final report.", tools=[save_report])
researcher = Agent(name="researcher", model=model, system_prompt="You research topics.",
                   tools=[web_search, writer.as_tool(), *EXTRA])


def main(topic: str = "x") -> str:
    return str(researcher(topic))
