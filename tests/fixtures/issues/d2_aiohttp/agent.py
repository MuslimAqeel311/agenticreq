"""An async aiohttp client to the Anthropic messages API."""

import aiohttp

TOOLS = [{"name": "lookup", "description": "Look up.",
          "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}]


async def main(task: str = "go") -> str:
    msgs = [{"role": "user", "content": task}]
    async with aiohttp.ClientSession() as s:
        for _ in range(3):
            async with s.post("https://api.anthropic.com/v1/messages",
                              json={"model": "claude-sonnet-4-5-20250929", "max_tokens": 9, "system": "be brief",
                                    "messages": msgs, "tools": TOOLS}) as resp:
                r = await resp.json()
            msgs.append({"role": "assistant", "content": r["content"]})
            uses = [b for b in r["content"] if b["type"] == "tool_use"]
            if not uses:
                return r["content"][0]["text"]
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"], "content": "hit"} for u in uses]})
    return ""
