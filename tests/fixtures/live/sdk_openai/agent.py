"""Plain openai SDK loop."""
import json

import openai

TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look something up.",
          "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]


def main(task: str = "go") -> str:
    client = openai.OpenAI(api_key="x")
    msgs = [{"role": "system", "content": "Help."}, {"role": "user", "content": task}]
    for _ in range(4):
        r = client.chat.completions.create(model="gpt-4o-2024-08-06", messages=msgs, tools=TOOLS)
        m = r.choices[0].message
        msgs.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            return m.content or ""
        for tc in m.tool_calls:
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"ok": 1})})
    return ""
