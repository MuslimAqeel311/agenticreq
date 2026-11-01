"""PydanticAI test agent: single agent, 4 plain-function tools, Anthropic, short model alias.

Ordinary user code -- knows nothing about the spike harness.
"""
from pydantic_ai import Agent

from shared_tools import SYSTEM_RULES, USER_TASK, get_inventory, pay_invoice, search_policy, submit_order


def main() -> str:
    agent = Agent("anthropic:claude-sonnet-4-5", system_prompt=SYSTEM_RULES,
                  tools=[get_inventory, search_policy, submit_order, pay_invoice])
    answer = agent.run_sync(USER_TASK).output
    print("AGENT OUTPUT:", answer)
    return answer


if __name__ == "__main__":
    main()
