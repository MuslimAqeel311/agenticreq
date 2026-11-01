"""A two-agent research team (google_adk): a researcher with a web search and tools loaded at runtime, a writer that
saves the report."""

import json
import os


def load_tool_specs():
    """Extra tools named in a config file: built at runtime, their names are not in the code."""
    with open(os.getenv("EXTRA_TOOLS", "extra_tools.json"), encoding="utf-8") as fh:
        return json.load(fh)


from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool


def web_search(query: str) -> str:
    """Search the web."""
    return query[:200]


def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text[:20000])
    return 'saved'


EXTRA = [FunctionTool(func=s["fn"]) for s in load_tool_specs()]
writer = LlmAgent(name="writer", model="gemini-2.0-flash-001", instruction="You write the final report.",
                  tools=[save_report])
researcher = LlmAgent(name="researcher", model="gemini-2.0-flash-001", instruction="You research topics.",
                      tools=[web_search, *EXTRA], sub_agents=[writer])
root_agent = researcher


def main(topic: str = "x") -> str:
    return topic
