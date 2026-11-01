"""No framework: one plain HTTP call per turn. The task picks which specialist's tools the model is offered, the way
a router hands one request to one specialist."""

import httpx

SPECIALISTS = {"billing": "get_invoice", "support": "open_ticket", "sales": "quote_price"}


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": name, "parameters": {
        "type": "object", "properties": {"ref": {"type": "string"}}, "required": ["ref"]}}}


def run(task: str = "billing") -> str:
    msgs: list = [{"role": "system", "content": f"You are the {task} specialist."}, {"role": "user", "content": task}]
    for _ in range(4):
        r = httpx.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": "Bearer sk-x"},
                       json={"model": "gpt-4o", "messages": msgs, "tools": [_tool(SPECIALISTS[task])]}).json()
        msg = r["choices"][0]["message"]
        msgs.append(msg)
        if not msg.get("tool_calls"):
            return str(msg["content"])
        for tc in msg["tool_calls"]:
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": "ok"})
    return "done"
