"""Triage -> Accounts / Fraud / Loans (OpenAI Agents): a handoff reaches ONE specialist per run."""

import asyncio
import sys

from agents import Agent, Runner, function_tool


@function_tool
def get_balances(customer_id: str) -> str:
    """Balances of a customer."""
    return "100"


@function_tool
def flag_transaction(txn_id: str) -> str:
    """Flag a transaction as suspicious."""
    return "flagged"


@function_tool
def loan_estimate(amount: float) -> str:
    """Estimate a loan."""
    return "5%"


accounts = Agent(name="Accounts", instructions="You answer account questions.", tools=[get_balances],
                 model="gpt-4.1-2025-04-14")
fraud = Agent(name="Fraud", instructions="You review suspicious activity.", tools=[flag_transaction],
              model="gpt-4.1-2025-04-14")
loans = Agent(name="Loans", instructions="You quote loans.", tools=[loan_estimate], model="gpt-4.1-2025-04-14")
triage = Agent(name="Triage", instructions="You route banking requests.", handoffs=[accounts, fraud, loans],
               model="gpt-4.1-2025-04-14")


async def run(task: str = "What is my balance?") -> None:
    await Runner.run(triage, task, max_turns=12)


if __name__ == "__main__":
    asyncio.run(run(*sys.argv[1:]))
