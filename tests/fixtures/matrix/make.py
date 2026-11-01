"""Writes the cross-framework matrix fixtures (one small multi-agent project per framework): two or more agents, a
write tool, a display name that differs from the name sent to the model, and a tool built at runtime.
Run it to regenerate: python tests/fixtures/matrix/make.py"""

from pathlib import Path

HERE = Path(__file__).parent
WRITE_BODY = ("    with open(path, 'w', encoding='utf-8') as fh:\n        fh.write(text[:20000])\n"
              "    return 'saved'\n")

PROJECTS = {
    "crewai": '''from crewai import Agent, Crew, Task
from crewai.tools import BaseTool, tool


class WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Search the web."

    def _run(self, query: str) -> str:
        return query[:200]


@tool("Save Report")
def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
''' + WRITE_BODY + '''

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
''',
    "langgraph": '''from langchain_core.tools import StructuredTool, tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor


@tool("web-search")
def web_search(query: str) -> str:
    """Search the web."""
    return query[:200]


@tool("save.report")
def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
''' + WRITE_BODY + '''

def make(spec):
    return StructuredTool.from_function(func=spec["fn"], name=spec["name"], description=spec["desc"])


EXTRA = [make(s) for s in load_tool_specs()]
researcher = create_react_agent("openai:gpt-4o-2024-08-06", [web_search, *EXTRA], name="researcher")
writer = create_react_agent("openai:gpt-4o-2024-08-06", [save_report], name="writer")
app = create_supervisor([researcher, writer], model="openai:gpt-4o-2024-08-06").compile()


def main(topic: str = "x") -> str:
    return str(app.invoke({"messages": [("user", topic)]}, config={"recursion_limit": 20}))
''',
    "openai_agents": '''from agents import Agent, FunctionTool, Runner, function_tool


@function_tool(name_override="web-search")
def web_search(query: str) -> str:
    """Search the web."""
    return query[:200]


@function_tool(name_override="Save Report")
def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
''' + WRITE_BODY + '''

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
''',
    "pydantic_ai": '''from pydantic_ai import Agent, RunContext, Tool


def web_search(query: str) -> str:
    """Search the web."""
    return query[:200]


def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
''' + WRITE_BODY + '''

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
''',
    "smolagents": '''from smolagents import OpenAIServerModel, Tool, ToolCallingAgent, tool


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
''' + WRITE_BODY + '''

EXTRA = [tool(s["fn"]) for s in load_tool_specs()]
model = OpenAIServerModel(model_id="gpt-4o-2024-08-06")
writer = ToolCallingAgent(tools=[save_report], model=model, name="writer", description="writes the report",
                          max_steps=6)
researcher = ToolCallingAgent(tools=[WebSearch(), *EXTRA], model=model, name="researcher",
                              description="researches", managed_agents=[writer], max_steps=6)


def main(topic: str = "x") -> str:
    return str(researcher.run(topic))
''',
    "autogen": '''from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def web_search(query: str) -> str:
    """Search the web."""
    return query[:200]


async def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
''' + WRITE_BODY + '''

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
''',
    "llamaindex": '''from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI


def web_search(query: str) -> str:
    """Search the web."""
    return query[:200]


def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
''' + WRITE_BODY + '''

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
''',
    "google_adk": '''from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool


def web_search(query: str) -> str:
    """Search the web."""
    return query[:200]


def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
''' + WRITE_BODY + '''

EXTRA = [FunctionTool(func=s["fn"]) for s in load_tool_specs()]
writer = LlmAgent(name="writer", model="gemini-2.0-flash-001", instruction="You write the final report.",
                  tools=[save_report])
researcher = LlmAgent(name="researcher", model="gemini-2.0-flash-001", instruction="You research topics.",
                      tools=[web_search, *EXTRA], sub_agents=[writer])
root_agent = researcher


def main(topic: str = "x") -> str:
    return topic
''',
    "agno": '''from agno.agent import Agent
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
''' + WRITE_BODY + '''

EXTRA = [tool(name=s["name"])(s["fn"]) for s in load_tool_specs()]
researcher = Agent(name="researcher", model=OpenAIChat(id="gpt-4o-2024-08-06"), instructions="You research topics.",
                   tools=[web_search, *EXTRA], tool_call_limit=10)
writer = Agent(name="writer", model=OpenAIChat(id="gpt-4o-2024-08-06"), instructions="You write the final report.",
               tools=[save_report], tool_call_limit=10)
team = Team(name="team", members=[researcher, writer], model=OpenAIChat(id="gpt-4o-2024-08-06"))


def main(topic: str = "x") -> str:
    return str(team.run(topic).content)
''',
    "strands": '''from strands import Agent, tool
from strands.models.openai import OpenAIModel


@tool(name="web-search")
def web_search(query: str) -> str:
    """Search the web."""
    return query[:200]


@tool(name="Save Report")
def save_report(path: str, text: str) -> str:
    """Save the report to a file."""
''' + WRITE_BODY + '''

EXTRA = [tool(name=s["name"])(s["fn"]) for s in load_tool_specs()]
model = OpenAIModel(model_id="gpt-4o-2024-08-06")
writer = Agent(name="writer", model=model, system_prompt="You write the final report.", tools=[save_report])
researcher = Agent(name="researcher", model=model, system_prompt="You research topics.",
                   tools=[web_search, writer.as_tool(), *EXTRA])


def main(topic: str = "x") -> str:
    return str(researcher(topic))
''',
}
HEADER = '''"""A two-agent research team ({fw}): a researcher with a web search and tools loaded at runtime, a writer that
saves the report."""

import json
import os


def load_tool_specs():
    """Extra tools named in a config file: built at runtime, their names are not in the code."""
    with open(os.getenv("EXTRA_TOOLS", "extra_tools.json"), encoding="utf-8") as fh:
        return json.load(fh)


'''

if __name__ == "__main__":
    for fw, code in PROJECTS.items():
        d = HERE / fw
        d.mkdir(exist_ok=True)
        (d / "agent.py").write_text(HEADER.format(fw=fw) + code, encoding="utf-8")
