"""Two helpdesk agents picked by name, like a supervisor's specialists."""

from helpdesk.minifw import Agent
from helpdesk.tools import BILLING_TOOLS, SUPPORT_TOOLS

TOOLSETS = {
    "support": SUPPORT_TOOLS,
    "billing": BILLING_TOOLS,
}


def build_agent(name: str) -> Agent:
    return Agent(name=name, tools=TOOLSETS[name])


def run_agent(name: str, task: str) -> str:
    agent = build_agent(name)
    return agent.invoke(task, max_turns=6)
