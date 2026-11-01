"""Three-agent OpenAI Agents chain: Triage → Policy Analyst → Payments."""

import asyncio

from agents import Agent, Runner, function_tool


@function_tool
def get_inventory(sku: str) -> str:
    """Look up stock for a SKU."""
    return "3 units"


@function_tool
def lookup_clause(clause_id: str) -> str:
    """Fetch a policy clause."""
    return "clause"


@function_tool
def pay_invoice(invoice_id: str, amount: float) -> str:
    """Pay a supplier invoice."""
    return "paid"


payments = Agent(name="Payments", instructions="You pay approved invoices.", tools=[pay_invoice],
                 model="gpt-4.1-2025-04-14")
analyst = Agent(name="Policy Analyst", instructions="You analyse procurement policy clauses.", tools=[lookup_clause],
                handoffs=[payments], model="gpt-4.1-2025-04-14")
triage = Agent(name="Procurement Triage", instructions="You handle procurement requests.", tools=[get_inventory],
               handoffs=[analyst], model="gpt-4.1-2025-04-14")


async def main() -> None:
    await Runner.run(triage, "Restock A-100", max_turns=12)


if __name__ == "__main__":
    asyncio.run(main())
