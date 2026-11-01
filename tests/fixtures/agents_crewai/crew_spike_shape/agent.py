"""Spike-shape regression fixture: the spike's CrewAI tool signatures, reconstructed from gaps.md #7.

Assumption (the spike's source was not available): every parameter carries a description,
written as a Google-style `Args:` docstring section.
"""

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool


@tool("submit_order")
def submit_order(sku: str, quantity: int, priority: str) -> str:
    """Submit a purchase order.

    Args:
        sku: Stock keeping unit.
        quantity: Units to order.
        priority: Order priority (normal or urgent).
    """
    return "PO-1"


@tool("pay_invoice")
def pay_invoice(invoice_id: str, amount: float) -> str:
    """Pay a supplier invoice.

    Args:
        invoice_id: Supplier invoice identifier.
        amount: Amount to pay.
    """
    return "paid"


@tool("get_inventory")
def get_inventory(sku: str, warehouse: str = "KHI") -> str:
    """Look up stock for a SKU.

    Args:
        sku: Stock keeping unit.
        warehouse: Warehouse code.
    """
    return "3 units"


@tool("search_policy")
def search_policy(query: str, max_results: int = 3) -> str:
    """Search the procurement policy.

    Args:
        query: Search text.
        max_results: Maximum passages to return.
    """
    return "policy text"


llm = LLM(model="gpt-4o")
buyer = Agent(role="Procurement Agent", goal="Restock low inventory", backstory="You run procurement.",
              tools=[submit_order, pay_invoice, get_inventory, search_policy], llm=llm, max_iter=8)
task = Task(description="Restock A-100 following policy", expected_output="PO number", agent=buyer)
crew = Crew(agents=[buyer], tasks=[task], process=Process.sequential)

if __name__ == "__main__":
    print(crew.kickoff())
