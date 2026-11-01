"""A two-agent research team (langgraph): a researcher with a web search and tools loaded at runtime, a writer that
saves the report."""

import json
import os


def load_tool_specs():
    """Extra tools named in a config file: built at runtime, their names are not in the code."""
    with open(os.getenv("EXTRA_TOOLS", "extra_tools.json"), encoding="utf-8") as fh:
        return json.load(fh)


from langchain_core.tools import StructuredTool, tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor


@tool("web-search")
def web_search(query: str) -> str:
    """Search the web."""
    return query[:200]


@tool("save.report")
def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text[:20000])
    return 'saved'


def make(spec):
    return StructuredTool.from_function(func=spec["fn"], name=spec["name"], description=spec["desc"])


EXTRA = [make(s) for s in load_tool_specs()]
researcher = create_react_agent("openai:gpt-4o-2024-08-06", [web_search, *EXTRA], name="researcher")
writer = create_react_agent("openai:gpt-4o-2024-08-06", [save_report], name="writer")
app = create_supervisor([researcher, writer], model="openai:gpt-4o-2024-08-06").compile()


def main(topic: str = "x") -> str:
    return str(app.invoke({"messages": [("user", topic)]}, config={"recursion_limit": 20}))
