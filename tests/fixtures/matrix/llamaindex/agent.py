"""A two-agent research team (llamaindex): a researcher with a web search and tools loaded at runtime, a writer that
saves the report."""

import json
import os


def load_tool_specs():
    """Extra tools named in a config file: built at runtime, their names are not in the code."""
    with open(os.getenv("EXTRA_TOOLS", "extra_tools.json"), encoding="utf-8") as fh:
        return json.load(fh)


from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI


def web_search(query: str) -> str:
    """Search the web."""
    return query[:200]


def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text[:20000])
    return 'saved'


EXTRA = [FunctionTool.from_defaults(fn=s["fn"], name=s["name"]) for s in load_tool_specs()]
llm = OpenAI(model="gpt-4o-2024-08-06")
researcher = FunctionAgent(name="researcher", description="researches", system_prompt="You research topics.",
                           tools=[FunctionTool.from_defaults(fn=web_search, name="Web Search"), *EXTRA], llm=llm,
                           can_handoff_to=["writer"])
writer = FunctionAgent(name="writer", description="writes", system_prompt="You write the final report.",
                       tools=[FunctionTool.from_defaults(fn=save_report, name="save-report")], llm=llm)
workflow = AgentWorkflow(agents=[researcher, writer], root_agent="researcher", timeout=120)


async def main(topic: str = "x") -> str:
    return str(await workflow.run(user_msg=topic))
