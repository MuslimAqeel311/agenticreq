"""Legacy client path: plain httpx (as used by openai<3 / anthropic<1) posting chat completions."""

import json
import os

import httpx

URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions"
TOOLS = [{"type": "function", "function": {
    "name": "get_inventory", "description": "Look up stock for a SKU.",
    "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}}}]


def main() -> None:
    msgs: list = [{"role": "system", "content": "Legacy agent"}, {"role": "user", "content": "stock A-100?"}]
    with httpx.Client() as c:
        for _ in range(4):
            r = c.post(URL, json={"model": "gpt-4o-mini", "messages": msgs, "tools": TOOLS},
                       headers={"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}).json()
            m = r["choices"][0]["message"]
            msgs.append(m)
            if not m.get("tool_calls"):
                break
            for tc in m["tool_calls"]:
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"units": 3})})
        # a non-LLM egress attempt that the sandbox must block
        try:
            c.get("https://erp.example.com/api/stock")
        except httpx.HTTPError as e:
            print("blocked:", e)


if __name__ == "__main__":
    main()
