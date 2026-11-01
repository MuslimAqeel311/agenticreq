"""LangGraph recognizer: static node/edge reachability → which ToolNode tools can run.

Parses StateGraph.add_node / add_edge / add_conditional_edges / set_entry_point. Does not
import the user's graph (``.get_graph()`` would require executing their code).
"""

from __future__ import annotations

import ast
from pathlib import Path

from agenticreq.ast.extractor import iter_py_files
from agenticreq.ast.schema import _name, literal
from agenticreq.models.observed import ObservedAgent


class LangGraphRecognizer:
    api_version = 1
    framework = "langgraph"

    def recognize(self, obs: ObservedAgent, root: Path) -> None:
        edges: dict[str, set[str]] = {}
        node_tools: dict[str, list[str]] = {}
        dynamic_from: set[str] = set()
        var_to_tool = {t.func_name or t.name: t.name for t in obs.declared_tools.values()} | obs.tool_vars
        lists: dict[str, list[str]] = {}
        for p in iter_py_files(root):
            try:
                tree = ast.parse(p.read_text(encoding="utf-8-sig", errors="replace"))
            except SyntaxError:
                continue
            for n in ast.walk(tree):
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.List):
                    lists[n.targets[0].id] = [var_to_tool.get(_name(e), _name(e)) for e in n.value.elts]
            for n in ast.walk(tree):
                if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Attribute):
                    continue
                m = n.func.attr
                args = n.args
                if m == "add_node" and args:
                    ok, name = literal(args[0])
                    if not ok:
                        name = _name(args[0])
                    tgt = args[1] if len(args) > 1 else None
                    if isinstance(tgt, ast.Call) and _name(tgt.func) == "ToolNode" and tgt.args:
                        a0 = tgt.args[0]
                        if isinstance(a0, ast.List):
                            node_tools[str(name)] = [var_to_tool.get(_name(e), _name(e)) for e in a0.elts]
                        elif isinstance(a0, ast.Name):
                            node_tools[str(name)] = lists.get(a0.id, [])
                    edges.setdefault(str(name), set())
                elif m == "add_edge" and len(args) >= 2:
                    a, b = (str(literal(x)[1]) if literal(x)[0] else _name(x) for x in args[:2])
                    edges.setdefault(a, set()).add(b)
                elif m == "add_conditional_edges" and len(args) >= 2:
                    src = str(literal(args[0])[1]) if literal(args[0])[0] else _name(args[0])
                    targets: set[str] = set()
                    path_map: ast.expr | None
                    if len(args) > 2:
                        path_map = args[2]
                    else:
                        path_map = next((k.value for k in n.keywords if k.arg == "path_map"), None)
                    ok, pm = literal(path_map) if path_map is not None else (False, None)
                    if ok and isinstance(pm, dict):
                        targets |= {str(v) for v in pm.values()}
                    elif ok and isinstance(pm, list):
                        targets |= {str(v) for v in pm}
                    elif _name(args[1]) == "tools_condition":
                        targets |= {"tools", "__end__"}
                    else:
                        dynamic_from.add(src)  # router function: any node may be next
                    edges.setdefault(src, set()).update(targets)
                elif m == "set_entry_point" and args:
                    edges.setdefault("__start__", set()).add(str(literal(args[0])[1]))
        if not node_tools:
            return
        # START may be spelled START / "__start__"
        for k in list(edges):
            if k in ("START",):
                edges.setdefault("__start__", set()).update(edges.pop(k))
        for k, vs in edges.items():
            edges[k] = {"__start__" if v == "START" else ("__end__" if v == "END" else v) for v in vs}
        seen: set[str] = set()
        stack = ["__start__"]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            nxt = set(edges.get(cur, set()))
            if cur in dynamic_from:
                nxt |= set(edges)  # unknown router: conservatively everything
            stack.extend(nxt - seen)
        reach: dict[str, bool] = {}
        for node, tools in node_tools.items():
            for t in tools:
                reach[t] = reach.get(t, False) or node in seen
        obs.graph_reachable = reach
