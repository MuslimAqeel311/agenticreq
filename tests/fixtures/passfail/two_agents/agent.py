"""Two independent agents, no handoff or delegation between them."""
from agents import Agent, Runner, function_tool


@function_tool
def lookup_order(order_id: str) -> str:
    """Return the status of an order."""
    return {"A1": "shipped"}.get(order_id, "unknown")


@function_tool
def save_report(text: str) -> str:
    """Write a report file."""
    with open("report.txt", "w") as fh:
        fh.write(text)
    return "saved"


support = Agent(name="support", instructions="Answer order questions.", tools=[lookup_order],
                model="gpt-4o-2024-08-06")
reporter = Agent(name="reporter", instructions="Write reports.", tools=[save_report],
                 model="gpt-4o-2024-08-06")


def main(question: str) -> str:
    Runner.run_sync(reporter, question, max_turns=5)
    return Runner.run_sync(support, question, max_turns=10).final_output
