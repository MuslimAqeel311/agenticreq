"""A two-agent research team (smolagents): a researcher with a web search and tools loaded at runtime, a writer that
saves the report."""

import json
import os


def load_tool_specs():
    """Extra tools named in a config file: built at runtime, their names are not in the code."""
    with open(os.getenv("EXTRA_TOOLS", "extra_tools.json"), encoding="utf-8") as fh:
        return json.load(fh)


from smolagents import OpenAIServerModel, Tool, ToolCallingAgent, tool


class WebSearch(Tool):
    name = "web-search"
    description = "Search the web."
    inputs = {"query": {"type": "string", "description": "the query"}}
    output_type = "string"

    def forward(self, query: str) -> str:
        return query[:200]


@tool
def save_report(path: str, text: str) -> str:
    """Save the report to a file.

    Args:
        path: where to save it.
        text: the report.
    """
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text[:20000])
    return 'saved'


EXTRA = [tool(s["fn"]) for s in load_tool_specs()]
model = OpenAIServerModel(model_id="gpt-4o-2024-08-06")
writer = ToolCallingAgent(tools=[save_report], model=model, name="writer", description="writes the report",
                          max_steps=6)
researcher = ToolCallingAgent(tools=[WebSearch(), *EXTRA], model=model, name="researcher",
                              description="researches", managed_agents=[writer], max_steps=6)


def main(topic: str = "x") -> str:
    return str(researcher.run(topic))
