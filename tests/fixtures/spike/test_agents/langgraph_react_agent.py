"""LangGraph test agent #1: prebuilt ReAct agent on Anthropic, 4 tools, pinned model.

Ordinary user code -- knows nothing about the spike harness.
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from shared_tools import SYSTEM_RULES, USER_TASK, get_inventory, pay_invoice, search_policy, submit_order


def main() -> str:
    tools = [tool(parse_docstring=True)(f) for f in (get_inventory, search_policy, submit_order, pay_invoice)]
    model = ChatAnthropic(model="claude-sonnet-4-5-20250929", max_tokens=1024)
    agent = create_react_agent(model, tools, prompt=SYSTEM_RULES)
    result = agent.invoke({"messages": [("user", USER_TASK)]})
    answer = result["messages"][-1].content
    print("AGENT OUTPUT:", answer)
    return answer


if __name__ == "__main__":
    main()
