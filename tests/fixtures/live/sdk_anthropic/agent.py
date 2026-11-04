"""Plain anthropic SDK loop."""
import anthropic

TOOLS = [{"name": "lookup", "description": "Look something up.",
          "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}]


def main(task: str = "go") -> str:
    client = anthropic.Anthropic(api_key="x")
    msgs = [{"role": "user", "content": task}]
    for _ in range(4):
        r = client.messages.create(model="claude-sonnet-4-5-20250929", max_tokens=64, system="Help.", messages=msgs, tools=TOOLS)
        msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in r.content]})
        uses = [b for b in r.content if b.type == "tool_use"]
        if not uses:
            return r.content[0].text
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": u.id, "content": "hit"} for u in uses]})
    return ""
