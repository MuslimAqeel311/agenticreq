"""smolagents ToolCallingAgent (schema-based)."""

from smolagents import OpenAIServerModel, ToolCallingAgent, tool


@tool
def get_inventory(sku: str, warehouse: str = "KHI") -> str:
    """Look up stock for a SKU.

    Args:
        sku: Stock keeping unit.
        warehouse: Warehouse code.
    """
    return "3 units"


@tool
def submit_order(sku: str, quantity: int) -> str:
    """Submit a purchase order.

    Args:
        sku: Stock keeping unit.
        quantity: Units to order.
    """
    return "PO-1"


model = OpenAIServerModel(model_id="gpt-4o")
agent = ToolCallingAgent(tools=[get_inventory, submit_order], model=model, max_steps=6)

if __name__ == "__main__":
    print(agent.run("Restock A-100"))
