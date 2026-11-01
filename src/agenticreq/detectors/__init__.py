"""Detector extension point: evidence that a prohibited *action* (not just a tool) happened.

A spec entry ``prohibited: [{action: "...", detector: {type: <name>, params: {...}}}]`` is checked by
the detector registered under ``<name>``. Detectors only read real evidence (non-synthetic captures,
traces) and return one evidence string per hit: they can find violations, never prove safety.

    class MyDetector:
        name = "my_detector"
        def detect(self, params: dict, obs: ObservedAgent) -> list[str]: ...

    register(MyDetector())
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from agenticreq.models.observed import ObservedAgent


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, params: dict[str, Any], obs: ObservedAgent) -> list[str]: ...


_REGISTRY: dict[str, Detector] = {}


def register(d: Detector) -> Detector:
    _REGISTRY[d.name] = d
    return d


def get(name: str) -> Detector | None:
    return _REGISTRY.get(name)


def _real_tool_calls(obs: ObservedAgent) -> list[tuple[str, str, dict[str, Any]]]:
    """(run_id, tool, args) for every tool call made by a real model."""
    out: list[tuple[str, str, dict[str, Any]]] = []
    for c in obs.captures:
        if not c.synthetic:
            out += [(c.run_id, tc.name, tc.arguments) for tc in c.tool_calls]
    for s in obs.trace_spans:
        if s.tool_name:
            raw = s.attributes.get("gen_ai.tool.call.arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw if isinstance(raw, dict) else {})
            except json.JSONDecodeError:
                args = {"_raw": raw}
            out.append((s.trace_id or "trace", s.tool_name, args if isinstance(args, dict) else {}))
    return out


class ToolArgRegex:
    """params: {tool: name (optional), arg: name (optional → all args), pattern: regex}"""

    name = "tool_arg_regex"

    def detect(self, params: dict[str, Any], obs: ObservedAgent) -> list[str]:
        pat = re.compile(str(params["pattern"]))
        tool, arg = params.get("tool"), params.get("arg")
        hits = []
        for run, name, args in _real_tool_calls(obs):
            if tool and name != tool:
                continue
            values = {arg: args.get(arg)} if arg else args
            for k, v in values.items():
                if v is not None and pat.search(json.dumps(v) if not isinstance(v, str) else v):
                    hits.append(f"{run}: {name}({k}={str(v)[:60]!r}) matches /{pat.pattern}/")
        return hits


register(ToolArgRegex())
