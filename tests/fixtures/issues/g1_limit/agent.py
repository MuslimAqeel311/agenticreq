"""The schema allows more than the spec's limit."""

from typing import Annotated

from agents import Agent, function_tool
from pydantic import Field


@function_tool
def transfer(amount: Annotated[int, Field(le=600)]) -> str:
    """Transfer an amount."""
    return "ok"


agent = Agent(name="payer", instructions="Pay.", tools=[transfer], model="gpt-4o-2024-08-06")
