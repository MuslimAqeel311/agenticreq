"""A tiny agent loop over the OpenAI chat API (httpx), so the fixture needs no framework installed."""

import inspect
import json
import os
from typing import Any, Callable

import httpx


class Tool:
    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func
        self.name = func.__name__
        params = inspect.signature(func).parameters
        types = {str: "string", float: "number", int: "integer"}
        self.schema = {"type": "function", "function": {
            "name": self.name, "description": (func.__doc__ or "").strip(),
            "parameters": {"type": "object", "required": list(params),
                           "properties": {n: {"type": types.get(p.annotation, "string")} for n, p in params.items()}}}}

    def __call__(self, **kw: Any) -> Any:
        return self.func(**kw)


def tool(func: Callable[..., Any]) -> Tool:
    return Tool(func)


class Agent:
    def __init__(self, name: str, tools: list[Tool], model: str = "gpt-4o-2024-08-06") -> None:
        self.name, self.tools, self.model = name, {t.name: t for t in tools}, model

    def invoke(self, task: str, max_turns: int = 6) -> str:
        url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions"
        msgs: list[dict[str, Any]] = [{"role": "system", "content": f"You are {self.name}."},
                                      {"role": "user", "content": task}]
        with httpx.Client() as c:
            for _ in range(max_turns):
                r = c.post(url, json={"model": self.model, "messages": msgs,
                                      "tools": [t.schema for t in self.tools.values()]},
                           headers={"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}).json()
                m = r["choices"][0]["message"]
                msgs.append(m)
                if not m.get("tool_calls"):
                    return str(m.get("content") or "")
                for tc in m["tool_calls"]:
                    fn = self.tools[tc["function"]["name"]]
                    out = fn.func(**json.loads(tc["function"]["arguments"] or "{}"))
                    msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": str(out)})
        return ""
