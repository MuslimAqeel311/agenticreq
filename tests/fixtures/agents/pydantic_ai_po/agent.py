"""PydanticAI agent on Anthropic (the beta.messages.create path S1 missed)."""

from pydantic_ai import Agent, RunContext

agent = Agent("anthropic:claude-sonnet-4-5", system_prompt="You are the procurement agent.", deps_type=str)


@agent.tool
def get_inventory(ctx: RunContext[str], sku: str, warehouse: str = "KHI") -> str:
    """Look up stock for a SKU.

    Args:
        sku: Stock keeping unit.
        warehouse: Warehouse code.
    """
    return "3 units"


@agent.tool_plain
def submit_order(sku: str, quantity: int) -> str:
    """Submit a purchase order.

    Args:
        sku: Stock keeping unit.
        quantity: Units to order.
    """
    return "PO-1"


if __name__ == "__main__":
    print(agent.run_sync("Restock A-100", deps="tenant-1").output)
