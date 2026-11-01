"""Support agent that is (wrongly) given a refund tool the spec prohibits."""
from agents import Agent, Runner, function_tool


@function_tool
def lookup_order(order_id: str) -> str:
    """Return the status of an order."""
    return {"A1": "shipped"}.get(order_id, "unknown")


@function_tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return f"refunded {amount} on {order_id}"


support = Agent(name="support", instructions="Answer order questions.", tools=[lookup_order, issue_refund],
                model="gpt-4o-2024-08-06")


def main(question: str) -> str:
    return Runner.run_sync(support, question, max_turns=10).final_output
