"""A raw-HTTP tool loop: the model may delete records, but only after a human types yes."""
import json

import httpx

TOOLS = [
    {"type": "function", "function": {"name": "search_records", "description": "Search records",
                                      "parameters": {"type": "object", "properties": {"q": {"type": "string"}},
                                                     "required": ["q"]}}},
    {"type": "function", "function": {"name": "delete_record", "description": "Delete a record",
                                      "parameters": {"type": "object", "properties": {"rid": {"type": "string"}},
                                                     "required": ["rid"]}}},
]


def search_records(q: str) -> str:
    return "r1"


def delete_record(rid: str) -> str:
    if input(f"Delete {rid}? [yes/no] ").strip().lower() != "yes":  # GATE
        return "denied by operator"
    return f"deleted {rid}"


IMPL = {"search_records": search_records, "delete_record": delete_record}


def main(task: str = "clean up") -> str:
    msgs = [{"role": "system", "content": "You manage records."}, {"role": "user", "content": task}]
    with httpx.Client(timeout=30) as client:
        for _ in range(8):
            r = client.post("https://api.openai.com/v1/chat/completions", headers={"authorization": "Bearer x"},
                            json={"model": "gpt-4o-2024-08-06", "messages": msgs, "tools": TOOLS}).json()
            msg = r["choices"][0]["message"]
            msgs.append(msg)
            if not msg.get("tool_calls"):
                return msg["content"]
            for tc in msg["tool_calls"]:
                out = IMPL[tc["function"]["name"]](**json.loads(tc["function"]["arguments"]))
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
    return "stopped"
