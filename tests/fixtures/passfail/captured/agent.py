"""A minimal OpenAI Agents SDK agent: one read-only tool, a pinned model, a step bound."""
from agents import Agent, Runner, function_tool


@function_tool
def lookup_order(order_id: str) -> str:
    """Return the status of an order."""
    return {"A1": "shipped"}.get(order_id, "unknown")


support = Agent(name="support", instructions="Answer order questions.", tools=[lookup_order],
                model="gpt-4o-2024-08-06")


def main(question: str) -> str:
    return Runner.run_sync(support, question, max_turns=10).final_output
