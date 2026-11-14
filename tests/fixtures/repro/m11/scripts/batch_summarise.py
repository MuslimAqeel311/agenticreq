"""A nightly batch script in the same folder: its own unpinned model and a delete tool. Not the shipped agent."""

from agents import Agent, Runner, function_tool


@function_tool
def delete_old_rows(days: int) -> str:
    """Delete rows older than some days."""
    return "deleted"


batch = Agent(name="batch", model="gpt-4o", tools=[delete_old_rows])


def nightly() -> None:
    Runner.run_sync(batch, "clean up")
