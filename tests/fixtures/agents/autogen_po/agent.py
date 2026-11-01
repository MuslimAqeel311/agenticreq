"""AutoGen AgentChat: a procurement assistant with two tools (OpenAI chat client)."""

import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


async def get_inventory(sku: str, warehouse: str = "KHI") -> str:
    """Look up stock for a SKU."""
    return "3 units"


async def submit_order(sku: str, quantity: int) -> str:
    """Submit a purchase order."""
    return "PO-1"


client = OpenAIChatCompletionClient(model="gpt-4o-2024-08-06", api_key="sk-placeholder")
buyer = AssistantAgent(name="buyer", model_client=client, tools=[get_inventory, submit_order],
                       system_message="You restock inventory.", reflect_on_tool_use=False, max_tool_iterations=5)


async def main() -> None:
    result = await buyer.run(task="Restock A-100")
    print(result.messages[-1].content)


if __name__ == "__main__":
    asyncio.run(main())
