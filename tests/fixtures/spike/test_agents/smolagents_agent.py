"""Smolagents test agent: ToolCallingAgent on an OpenAI-compatible model, pinned model id.

Ordinary user code -- knows nothing about the spike harness.
"""
from smolagents import OpenAIServerModel, ToolCallingAgent, tool

from shared_tools import SYSTEM_RULES, USER_TASK, get_inventory, pay_invoice, search_policy, submit_order


def main() -> str:
    model = OpenAIServerModel(model_id="gpt-4o-mini-2024-07-18")
    agent = ToolCallingAgent(
        tools=[tool(f) for f in (get_inventory, search_policy, submit_order, pay_invoice)],
        model=model, instructions=SYSTEM_RULES, max_steps=6, verbosity_level=0,
    )
    answer = str(agent.run(USER_TASK))
    print("AGENT OUTPUT:", answer)
    return answer


if __name__ == "__main__":
    main()
