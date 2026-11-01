"""A two-agent research team (autogen): a researcher with a web search and tools loaded at runtime, a writer that
saves the report."""

import json
import os


def load_tool_specs():
    """Extra tools named in a config file: built at runtime, their names are not in the code."""
    with open(os.getenv("EXTRA_TOOLS", "extra_tools.json"), encoding="utf-8") as fh:
        return json.load(fh)


from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def web_search(query: str) -> str:
    """Search the web."""
    return query[:200]


async def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text[:20000])
    return 'saved'


EXTRA = [FunctionTool(s["fn"], name=s["name"], description=s["desc"]) for s in load_tool_specs()]
client = OpenAIChatCompletionClient(model="gpt-4o-2024-08-06")
researcher = AssistantAgent(name="researcher", model_client=client, system_message="You research topics.",
                            tools=[FunctionTool(web_search, name="Web-Search", description="Search the web."), *EXTRA],
                            max_tool_iterations=5)
writer = AssistantAgent(name="writer", model_client=client, system_message="You write the final report.",
                        tools=[FunctionTool(save_report, name="save.report", description="Save the report.")],
                        max_tool_iterations=5)
team = RoundRobinGroupChat([researcher, writer], max_turns=6)


async def main(topic: str = "x") -> str:
    return str(await team.run(task=topic))
