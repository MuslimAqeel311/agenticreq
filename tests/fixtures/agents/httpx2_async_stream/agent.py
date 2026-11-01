"""Async + streaming: AsyncOpenAI (httpx2 under openai>=3) with stream=True."""

import asyncio

from openai import AsyncOpenAI

TOOLS = [{"type": "function", "function": {
    "name": "submit_order", "description": "Submit a purchase order.",
    "parameters": {"type": "object", "properties": {"sku": {"type": "string"}, "quantity": {"type": "integer"}},
                   "required": ["sku", "quantity"]}}}]


async def main() -> None:
    client = AsyncOpenAI()
    msgs: list = [{"role": "system", "content": "Streaming agent"}, {"role": "user", "content": "order A-100"}]
    for _ in range(4):
        stream = await client.chat.completions.create(model="gpt-4o-2024-08-06", messages=msgs, tools=TOOLS,
                                                      stream=True)
        calls: dict = {}
        text = ""
        async for chunk in stream:
            d = chunk.choices[0].delta if chunk.choices else None
            if d and d.tool_calls:
                for tc in d.tool_calls:
                    calls.setdefault(tc.index, {"id": tc.id, "name": tc.function.name, "args": ""})
                    calls[tc.index]["args"] += tc.function.arguments or ""
            if d and d.content:
                text += d.content
        if not calls:
            break
        msgs.append({"role": "assistant", "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"]}}
            for c in calls.values()]})
        for c in calls.values():
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": "PO-1"})


if __name__ == "__main__":
    asyncio.run(main())
