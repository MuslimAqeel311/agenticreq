"""Support agent whose model name is read from an environment variable."""
import os

from agents import Agent, Runner, function_tool


@function_tool
def lookup_order(order_id: str) -> str:
    """Return the status of an order."""
    return {"A1": "shipped"}.get(order_id, "unknown")


support = Agent(name="support", instructions="Answer order questions.", tools=[lookup_order],
                model=os.environ["SUPPORT_MODEL"])


def main(question: str) -> str:
    return Runner.run_sync(support, question, max_turns=10).final_output
