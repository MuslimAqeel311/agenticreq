"""Raw Anthropic SDK tool loop on beta.messages.create (the path S1 missed)."""

import anthropic

MODEL = "claude-sonnet-4-5"
TOOLS = [
    {"name": "get_inventory", "description": "Look up stock for a SKU.",
     "input_schema": {"type": "object", "properties": {"sku": {"type": "string", "description": "SKU"}},
                      "required": ["sku"]}},
    {"name": "pay_invoice", "description": "Pay a supplier invoice.",
     "input_schema": {"type": "object", "properties": {"invoice_id": {"type": "string"},
                                                       "amount": {"type": "number", "maximum": 5000}},
                      "required": ["invoice_id", "amount"]}},
]


def run_tool(name: str, args: dict) -> str:
    return f"{name} ok"


def main() -> None:
    client = anthropic.Anthropic()
    messages: list = [{"role": "user", "content": "Restock and pay INV-1"}]
    for _ in range(8):
        resp = client.beta.messages.create(model=MODEL, max_tokens=512, system="You are the AP agent.",
                                           tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": [b.model_dump() for b in resp.content]})
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            break
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": u.id, "content": run_tool(u.name, u.input)} for u in uses]})


if __name__ == "__main__":
    main()
