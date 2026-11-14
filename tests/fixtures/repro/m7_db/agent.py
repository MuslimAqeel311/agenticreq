"""A read tool that queries a real database, and an engine opened at import."""
import requests
from agents import Agent, Runner, function_tool
from sqlalchemy import create_engine, text

engine = create_engine("postgresql://user@db.internal/prod")


@function_tool
def count_orders(day: str) -> str:
    """Count the orders of one day."""
    with engine.connect() as c:
        return str(c.execute(text("select count(*) from orders where day = :d"), {"d": day}).scalar())


@function_tool
def fetch_rate(currency: str) -> str:
    """Get an exchange rate."""
    return requests.get(f"https://rates.example.com/{currency}", timeout=5).text


agent = Agent(name="ops", model="gpt-4o-2024-08-06", tools=[count_orders, fetch_rate])


def main(task: str) -> str:
    return str(Runner.run_sync(agent, task, max_turns=6).final_output)
