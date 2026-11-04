"""Limits written in the idioms the library did not read."""

from dataclasses import dataclass

from crewai import Agent as CrewAgent
from pydantic_ai import Agent

DRY_RUN = True


@dataclass
class Guard:
    max_no_progress_steps: int = 5


def send_invoice(ref: str) -> str:
    """Send the invoice to the customer."""
    if DRY_RUN:
        return f"would send {ref}"
    import requests

    requests.post("https://billing.example.com/send", json={"ref": ref})
    return "sent"


agent = Agent("openai:gpt-4o-2024-08-06", retries=3, output_retries=2, tools=[send_invoice])
crew_agent = CrewAgent(
    role="biller", goal="bill", backstory="b", llm="gpt-4o-2024-08-06",
    max_retry_limit=4, max_rpm=10, max_execution_time=120,
)
