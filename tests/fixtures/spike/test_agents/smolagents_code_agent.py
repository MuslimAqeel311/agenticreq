"""Smolagents test agent #2: CodeAgent (smolagents' default agent type).

CodeAgent asks the model to write Python that calls the tools; tools are described in the
system prompt as Python signatures. Ordinary user code -- knows nothing about the spike harness.
"""
from smolagents import CodeAgent, OpenAIServerModel, tool

from shared_tools import SYSTEM_RULES, USER_TASK, get_inventory, pay_invoice, search_policy, submit_order


def main() -> str:
    model = OpenAIServerModel(model_id="gpt-4o-mini-2024-07-18")
    agent = CodeAgent(
        tools=[tool(f) for f in (get_inventory, search_policy, submit_order, pay_invoice)],
        model=model, instructions=SYSTEM_RULES, max_steps=6, verbosity_level=0,
    )
    answer = str(agent.run(USER_TASK))
    print("AGENT OUTPUT:", answer)
    return answer


if __name__ == "__main__":
    main()
