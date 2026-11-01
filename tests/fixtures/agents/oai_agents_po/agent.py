"""OpenAI Agents SDK — triage agent with a handoff to a policy analyst (responses.create)."""

import asyncio

from agents import Agent, Runner, function_tool


@function_tool
def get_inventory(sku: str, warehouse: str = "KHI") -> str:
    """Look up stock for a SKU.

    Args:
        sku: Stock keeping unit.
        warehouse: Warehouse code.
    """
    return "3 units"


@function_tool
def submit_order(sku: str, quantity: int) -> str:
    """Submit a purchase order."""
    return "PO-1"


@function_tool
def lookup_clause(clause_id: str, max_results: int = 5) -> str:
    """Fetch a policy clause.

    Args:
        clause_id: Clause identifier.
        max_results: Maximum clauses to return.
    """
    return "clause text"


policy_analyst = Agent(name="Policy Analyst", instructions="You analyse procurement policy clauses.",
                       tools=[lookup_clause], model="gpt-4.1")
triage = Agent(name="Procurement Triage", instructions="You handle procurement requests. Hand policy questions off.",
               tools=[get_inventory, submit_order], handoffs=[policy_analyst], model="gpt-4.1")


async def main() -> None:
    result = await Runner.run(triage, "Restock A-100 and check the policy", max_turns=10)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
