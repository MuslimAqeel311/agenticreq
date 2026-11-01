"""smolagents CodeAgent — prompt-only execution model (tools live in the system prompt)."""

from smolagents import CodeAgent, OpenAIServerModel, tool


@tool
def get_inventory(sku: str) -> str:
    """Look up stock for a SKU.

    Args:
        sku: Stock keeping unit.
    """
    return "3 units"


model = OpenAIServerModel(model_id="gpt-4o")
agent = CodeAgent(tools=[get_inventory], model=model, max_steps=3)

if __name__ == "__main__":
    print(agent.run("How many A-100 do we have?"))
