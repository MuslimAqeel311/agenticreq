"""CrewAI procurement crew: manager delegates to a policy analyst (allow_delegation)."""

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool


@tool("get_inventory")
def get_inventory(sku: str, warehouse: str = "KHI", max_results: int = 50) -> str:
    """Look up stock for a SKU.

    Args:
        sku: Stock keeping unit, e.g. A-100.
        warehouse: Warehouse code.
        max_results: Maximum rows to return.
    """
    return "3 units"


@tool("submit_order")
def submit_order(sku: str, quantity: int, notes: str = "") -> str:
    """Submit a purchase order.

    Args:
        sku: Stock keeping unit.
        quantity: Units to order.
        notes: Free-text notes for the supplier.
    """
    return "PO-1"


@tool("lookup_clause")
def lookup_clause(clause_id: str, section: str, max_results: int = 5) -> str:
    """Fetch a policy clause.

    Args:
        clause_id: Clause identifier.
        section: Policy section name.
        max_results: Maximum clauses to return.
    """
    return "clause"


llm = LLM(model="gpt-4o")
manager = Agent(role="Procurement Manager", goal="Restock low inventory", backstory="You run procurement.",
                tools=[get_inventory, submit_order], allow_delegation=True, llm=llm, max_iter=5)
analyst = Agent(role="Policy Analyst", goal="Answer policy questions", backstory="You know the policy.",
                tools=[lookup_clause], allow_delegation=False, llm=llm, max_iter=5)
task = Task(description="Restock A-100 following policy", expected_output="PO number", agent=manager)
crew = Crew(agents=[manager, analyst], tasks=[task], process=Process.sequential)

if __name__ == "__main__":
    print(crew.kickoff())
