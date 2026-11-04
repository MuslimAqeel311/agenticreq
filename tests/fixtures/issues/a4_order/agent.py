"""A router: the model picks one specialist tool per task. Each tool records that it ran."""

from minifw import Agent, tool


@tool
def transfer_to_billing(note: str) -> str:
    """Hand over to billing."""
    return "billing"


@tool
def transfer_to_support(note: str) -> str:
    """Hand over to support."""
    return "support"


def main(task: str = "route") -> str:
    return Agent(name="router", tools=[transfer_to_billing, transfer_to_support]).invoke(task, max_turns=6)
