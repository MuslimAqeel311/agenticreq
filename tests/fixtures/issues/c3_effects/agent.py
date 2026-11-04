"""Tools whose body agenticreq cannot follow. spec.yml says what one of them does."""

import stripe
from agents import Agent, function_tool


@function_tool
def charge(ref: str) -> str:
    """Bill a customer."""
    return str(stripe.Charge.create(amount=1, source=ref))


@function_tool
def lookup(ref: str) -> str:
    """Bill details."""
    return str(stripe.Thing.get(ref))


agent = Agent(name="biller", instructions="Bill.", tools=[charge, lookup], model="gpt-4o-2024-08-06")
