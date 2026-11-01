"""A two-agent research team (openai_agents): a researcher with a web search and tools loaded at runtime, a writer that
saves the report."""

import json
import os


def load_tool_specs():
    """Extra tools named in a config file: built at runtime, their names are not in the code."""
    with open(os.getenv("EXTRA_TOOLS", "extra_tools.json"), encoding="utf-8") as fh:
        return json.load(fh)


from agents import Agent, FunctionTool, Runner, function_tool


@function_tool(name_override="web-search")
def web_search(query: str) -> str:
    """Search the web."""
    return query[:200]


@function_tool(name_override="Save Report")
def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text[:20000])
    return 'saved'


def make(spec):
    return FunctionTool(name=spec["name"], description=spec["desc"], params_json_schema=spec["schema"],
                        on_invoke_tool=spec["fn"])


EXTRA = [make(s) for s in load_tool_specs()]
writer = Agent(name="writer", instructions="You write the final report.", tools=[save_report],
               model="gpt-4o-2024-08-06")
researcher = Agent(name="researcher", instructions="You research topics, then hand off to the writer.",
                   tools=[web_search, *EXTRA], handoffs=[writer], model="gpt-4o-2024-08-06")


def main(topic: str = "x") -> str:
    return Runner.run_sync(researcher, topic, max_turns=10).final_output
