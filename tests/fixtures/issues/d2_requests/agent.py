"""A plain requests client to an OpenAI-compatible host."""

import requests

TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up.",
          "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]


def main(task: str = "go") -> str:
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": task}]
    for _ in range(3):
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          json={"model": "llama-3.3-70b-versatile", "messages": msgs, "tools": TOOLS}, timeout=5).json()
        m = r["choices"][0]["message"]
        msgs.append(m)
        if not m.get("tool_calls"):
            return m["content"]
        for tc in m["tool_calls"]:
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": "hit"})
    return ""
