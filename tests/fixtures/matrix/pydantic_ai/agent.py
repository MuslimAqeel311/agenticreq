"""A two-agent research team (pydantic_ai): a researcher with a web search and tools loaded at runtime, a writer that
saves the report."""

import json
import os


def load_tool_specs():
    """Extra tools named in a config file: built at runtime, their names are not in the code."""
    with open(os.getenv("EXTRA_TOOLS", "extra_tools.json"), encoding="utf-8") as fh:
        return json.load(fh)


from pydantic_ai import Agent, RunContext, Tool


def web_search(query: str) -> str:
    """Search the web."""
    return query[:200]


def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text[:20000])
    return 'saved'


EXTRA = [Tool(s["fn"], name=s["name"]) for s in load_tool_specs()]
writer = Agent("openai:gpt-4o-2024-08-06", name="writer", system_prompt="You write the final report.",
               tools=[Tool(save_report, name="save-report")])
researcher = Agent("openai:gpt-4o-2024-08-06", name="researcher", system_prompt="You research topics.",
                   tools=[Tool(web_search, name="Web Search"), *EXTRA])


@researcher.tool
async def ask_writer(ctx: RunContext[None], notes: str) -> str:
    """Hand the notes to the writer."""
    return (await writer.run(notes, usage=ctx.usage)).output


def main(topic: str = "x") -> str:
    return researcher.run_sync(topic).output
