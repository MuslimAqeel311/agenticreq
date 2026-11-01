"""OpenAI Agents SDK test agent: manager hands off to analyst; manager has a local input guardrail.

Ordinary user code -- knows nothing about the spike harness.
"""
from agents import Agent, GuardrailFunctionOutput, Runner, function_tool, input_guardrail

from shared_tools import SYSTEM_RULES, USER_TASK, get_inventory, pay_invoice, search_policy, submit_order


@input_guardrail
def no_huge_orders(ctx, agent, user_input) -> GuardrailFunctionOutput:
    """Pure-Python policy check. Runs client-side; never reaches the LLM."""
    return GuardrailFunctionOutput(output_info={"checked": True}, tripwire_triggered="1000 units" in str(user_input))


def main() -> str:
    analyst = Agent(
        name="Policy Analyst",
        instructions="You look up inventory and company policy.\n" + SYSTEM_RULES,
        model="gpt-4.1-2025-04-14",
        tools=[function_tool(get_inventory), function_tool(search_policy)],
    )
    manager = Agent(
        name="Procurement Manager",
        instructions="Handle procurement requests. Hand research questions to the Policy Analyst.\n" + SYSTEM_RULES,
        model="gpt-4.1-2025-04-14",
        tools=[function_tool(submit_order), function_tool(pay_invoice)],
        handoffs=[analyst],
        input_guardrails=[no_huge_orders],
    )
    answer = Runner.run_sync(manager, USER_TASK).final_output
    print("AGENT OUTPUT:", answer)
    return answer


if __name__ == "__main__":
    main()
