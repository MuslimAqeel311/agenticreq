"""Plain OpenAI SDK loop; save_record refuses unless a project-specific helper approves."""
import json
from pathlib import Path

from openai import OpenAI

client = OpenAI()


def _approved(action: str) -> bool:
    return input(f"approve {action}? ").strip() == "y"


def save_record(text: str) -> str:
    """Save a record to disk."""
    if not _approved("save_record"):
        return "not approved"
    Path("record.txt").write_text(text)
    return "saved"


TOOLS = [{"type": "function", "function": {"name": "save_record", "description": "Save a record",
          "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}}}]


def run(task: str) -> str:
    msgs = [{"role": "user", "content": task}]
    for _ in range(6):
        r = client.chat.completions.create(model="gpt-4o-2024-08-06", messages=msgs, tools=TOOLS)
        m = r.choices[0].message
        if not m.tool_calls:
            return m.content or ""
        msgs.append(m)
        for tc in m.tool_calls:
            out = save_record(**json.loads(tc.function.arguments))
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return ""
