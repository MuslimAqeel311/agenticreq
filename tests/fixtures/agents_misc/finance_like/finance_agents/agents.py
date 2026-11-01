from langgraph.prebuilt import create_react_agent

from .llm import get_llm
from .tools import MARKET_TOOLS, PLANNER_TOOLS, TRADER_TOOLS

PROMPTS = {"market_analyst": "Analyse markets.", "financial_planner": "Plan finances.", "trader": "Trade."}
TOOLSETS = {
    "market_analyst": MARKET_TOOLS,
    "financial_planner": PLANNER_TOOLS,
    "trader": TRADER_TOOLS,
}


def build_agent(name: str):
    return create_react_agent(get_llm(), TOOLSETS[name], prompt=PROMPTS[name], name=name)


def run_agent(name: str, task: str) -> str:
    agent = build_agent(name)
    result = agent.invoke({"messages": [("user", task)]}, {"recursion_limit": 25})
    return result["messages"][-1].content
