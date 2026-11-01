"""`agenticreq fix` — structural autofix patches. --dry-run is the default; nothing is written
unless --apply is passed explicitly.

Patches (one per finding kind):
* AS-208  add a step cap (max_iter / max_steps / max_turns / recursion_limit)
* AS-209  lower a code bound to the spec bound
* AS-210  replace a model alias with a dated snapshot (data/model_pins.yml)
* AS-211  add an `idempotency_key` parameter to write tools (body wiring left as a TODO)
"""

from __future__ import annotations

import ast
import difflib
from dataclasses import dataclass, field
from pathlib import Path

from agenticreq.ast.classify import (
    canonical,
    has_idempotency,
    is_test_file,
    is_write_tool,
    offers_snapshots,
    strip_provider,
)
from agenticreq.ast.extractor import iter_py_files
from agenticreq.ast.schema import _name
from agenticreq.data import load_yaml
from agenticreq.models.observed import ObservedAgent
from agenticreq.pathutil import relative

_STEP_KW = {"Agent": "max_iter", "CodeAgent": "max_steps", "ToolCallingAgent": "max_steps"}
_RUNNER_KW = {"run": "max_turns", "run_sync": "max_turns", "run_streamed": "max_turns"}
_STEP_KINDS = {"recursion_limit", "max_iter", "max_iterations", "max_turns", "max_steps"}
#: calls that build a runnable agent / graph: only `.invoke()` on these runs a loop that recursion_limit bounds
_GRAPH_BUILDERS = {
    "compile",
    "create_react_agent",
    "create_agent",
    "create_supervisor",
    "create_swarm",
    "create_tool_calling_agent",
    "AgentExecutor",
}


def _graph_vars(tree: ast.AST, obs: ObservedAgent) -> tuple[set[str], set[str]]:
    """(names holding a graph / agent, functions returning one) in this file."""
    names = {a.var for a in obs.declared_agents if a.var}
    builder_fns: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            isinstance(r, ast.Return) and isinstance(r.value, ast.Call) and _name(r.value.func) in _GRAPH_BUILDERS
            for r in ast.walk(n)
        ):
            builder_fns.add(n.name)  # def build(): return g.compile(...)
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and (_name(n.value.func) in _GRAPH_BUILDERS | builder_fns)
        ):
            names |= {_name(t) for t in n.targets if isinstance(t, (ast.Name, ast.Attribute))}
    return names, builder_fns


def _runs_graph(call: ast.Call, graphs: set[str], builders: set[str], tools: set[str]) -> bool:
    """`graph.invoke(x)`, `self.app.invoke(x)`, `build().invoke(x)` — not `tool.invoke(x)` / `llm.invoke(x)`."""
    recv = call.func.value if isinstance(call.func, ast.Attribute) else None
    if isinstance(recv, ast.Call):
        return _name(recv.func) in _GRAPH_BUILDERS | builders
    name = _name(recv) if recv is not None else ""
    return bool(name) and name in graphs and name not in tools


@dataclass
class Edit:
    line: int  # 1-based
    col: int  # 0-based
    end_line: int
    end_col: int
    text: str


@dataclass
class Patch:
    rule_id: str
    file: Path
    description: str
    edits: list[Edit] = field(default_factory=list)


def _offset(lines: list[str], line: int, col: int) -> int:
    return sum(len(x) for x in lines[: line - 1]) + col


def apply_edits(src: str, edits: list[Edit]) -> str:
    lines = src.splitlines(keepends=True)
    for e in sorted(edits, key=lambda e: (e.line, e.col), reverse=True):
        a, b = _offset(lines, e.line, e.col), _offset(lines, e.end_line, e.end_col)
        src = src[:a] + e.text + src[b:]
        lines = src.splitlines(keepends=True)
    return src


def _insert_kwarg(call: ast.Call, text: str) -> Edit:
    nodes: list[ast.expr] = [*call.args, *(k.value for k in call.keywords)]
    if nodes:
        last = max(nodes, key=lambda n: (n.end_lineno or 0, n.end_col_offset or 0))
        ln, col = last.end_lineno or call.lineno, last.end_col_offset or 0
        return Edit(ln, col, ln, col, f", {text}")
    ln, col = call.end_lineno or call.lineno, (call.end_col_offset or 1) - 1
    return Edit(ln, col, ln, col, text)


def plan(obs: ObservedAgent, root: Path, max_steps: int | None) -> list[Patch]:
    """Mechanical patches. ``max_steps`` is the number the user wrote in spec.yml (bounds.max_steps); without one no
    step-cap patch is made: agenticreq never picks a number to write into your code."""
    pins: dict[str, str] = (load_yaml("model_pins.yml") or {}).get("pins", {})
    write_fns = {
        t.func_name for t in obs.declared_tools.values() if is_write_tool(t) and not has_idempotency(t) and t.func_name
    }
    has_step = {b.code_ref.file for b in obs.bounds if b.kind in _STEP_KINDS and b.code_ref}
    patches: list[Patch] = []
    base = root if root.is_dir() else root.parent
    tools = set(obs.tool_vars) | {t.func_name for t in obs.declared_tools.values() if t.func_name}
    # a model served by a provider with no snapshots has nothing to pin to (AS-210 says so instead)
    unpinnable = {m.name for m in obs.models if not offers_snapshots(m.provider)}
    for path in iter_py_files(root):
        rel = str(relative(path, base) or path)
        if is_test_file(rel):
            continue  # tests configure agents too, but they do not ship: never patch them
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        graphs, builders = _graph_vars(tree, obs)
        for node in ast.walk(tree):
            # AS-210 model pinning
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and strip_provider(node.value) in pins
                and node.value not in unpinnable
            ):
                bare = strip_provider(node.value)
                pinned = node.value[: len(node.value) - len(bare)] + pins[bare]
                patches.append(
                    Patch(
                        "AS-210",
                        path,
                        f"{rel}:{node.lineno} pin {node.value} → {pinned}",
                        [
                            Edit(
                                node.lineno,
                                node.col_offset,
                                node.end_lineno or node.lineno,
                                node.end_col_offset or 0,
                                '"' + pinned + '"',
                            )
                        ],
                    )
                )
            if isinstance(node, ast.Call) and max_steps is not None:
                fname = _name(node.func)
                kws = {k.arg for k in node.keywords}
                # AS-208 step caps
                if rel not in has_step:
                    kw = _STEP_KW.get(fname)
                    # CrewAI's Agent takes max_iter; the OpenAI Agents SDK's Agent does not (Runner.run's
                    # max_turns bounds it), so a bare `Agent` is patched only with CrewAI's required role=
                    crew = fname != "Agent" or "role" in kws
                    if kw and crew and (kws & {"tools", "role", "model"}) and kw not in kws:
                        patches.append(
                            Patch(
                                "AS-208",
                                path,
                                f"{rel}:{node.lineno} add {kw}={max_steps} to {fname}()",
                                [_insert_kwarg(node, f"{kw}={max_steps}")],
                            )
                        )
                    if (
                        fname in _RUNNER_KW
                        and _name(node.func.value if isinstance(node.func, ast.Attribute) else node.func) == "Runner"
                        and "max_turns" not in kws
                    ):
                        patches.append(
                            Patch(
                                "AS-208",
                                path,
                                f"{rel}:{node.lineno} add max_turns={max_steps} to Runner.{fname}()",
                                [_insert_kwarg(node, f"max_turns={max_steps}")],
                            )
                        )
                    if (
                        fname in ("invoke", "ainvoke", "stream", "astream")
                        and "config" not in kws
                        and "langgraph" in obs.frameworks
                        and len(node.args) == 1
                        and _runs_graph(node, graphs, builders, tools)
                    ):
                        patches.append(
                            Patch(
                                "AS-208",
                                path,
                                f"{rel}:{node.lineno} add recursion_limit={max_steps}",
                                [_insert_kwarg(node, f'config={{"recursion_limit": {max_steps}}}')],
                            )
                        )
                # AS-209 lower a bound
                for k in node.keywords:
                    if (
                        k.arg in _STEP_KINDS
                        and isinstance(k.value, ast.Constant)
                        and isinstance(k.value.value, int)
                        and k.value.value > max_steps
                    ):
                        kv = k.value
                        patches.append(
                            Patch(
                                "AS-209",
                                path,
                                f"{rel}:{kv.lineno} {k.arg} {kv.value!r} → {max_steps}",
                                [
                                    Edit(
                                        kv.lineno,
                                        kv.col_offset,
                                        kv.end_lineno or kv.lineno,
                                        kv.end_col_offset or 0,
                                        str(max_steps),
                                    )
                                ],
                            )
                        )
            if isinstance(node, ast.Dict) and max_steps is not None:
                for dk, v in zip(node.keys, node.values):
                    if (
                        isinstance(dk, ast.Constant)
                        and dk.value == "recursion_limit"
                        and isinstance(v, ast.Constant)
                        and isinstance(v.value, int)
                        and v.value > max_steps
                    ):
                        patches.append(
                            Patch(
                                "AS-209",
                                path,
                                f"{rel}:{v.lineno} recursion_limit {v.value!r} → {max_steps}",
                                [
                                    Edit(
                                        v.lineno,
                                        v.col_offset,
                                        v.end_lineno or v.lineno,
                                        v.end_col_offset or 0,
                                        str(max_steps),
                                    )
                                ],
                            )
                        )
            # AS-211 idempotency key
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in write_fns:
                a = node.args
                params = [*a.posonlyargs, *a.args, *a.kwonlyargs]
                if any(canonical(p.arg) == "idempotency_key" for p in params) or a.vararg or a.kwarg:
                    continue
                last = params[-1] if params else None
                if last is None:
                    continue
                end = last.annotation if last.annotation is not None else last
                defaults = [*a.defaults, *(d for d in a.kw_defaults if d is not None)]
                tail = max([end, *defaults], key=lambda n: (n.end_lineno or 0, n.end_col_offset or 0))
                ln, col = tail.end_lineno or node.lineno, tail.end_col_offset or 0
                text = ", idempotency_key: str | None = None"
                body0 = node.body[0]
                doc_skip = isinstance(body0, ast.Expr) and isinstance(getattr(body0, "value", None), ast.Constant)
                anchor = node.body[1] if doc_skip and len(node.body) > 1 else body0
                indent = " " * anchor.col_offset
                todo = (
                    f"{indent}# TODO(agenticreq AS-211): pass idempotency_key to the side-effecting call "
                    "so retries cannot repeat it\n"
                )
                patches.append(
                    Patch(
                        "AS-211",
                        path,
                        f"{rel}:{node.lineno} add idempotency_key to {node.name}()",
                        [Edit(ln, col, ln, col, text), Edit(anchor.lineno, 0, anchor.lineno, 0, todo)],
                    )
                )
    return patches


def render_diff(patches: list[Patch], base: Path) -> str:
    out: list[str] = []
    by_file: dict[Path, list[Patch]] = {}
    for p in patches:
        by_file.setdefault(p.file, []).append(p)
    for f, ps in by_file.items():
        src = f.read_text(encoding="utf-8")
        new = apply_edits(src, [e for p in ps for e in p.edits])
        r = relative(f, base)
        rel = (r or f).as_posix()  # diff headers use forward slashes on every OS, as git does
        out += difflib.unified_diff(
            src.splitlines(keepends=True), new.splitlines(keepends=True), fromfile=f"a/{rel}", tofile=f"b/{rel}"
        )
    return "".join(out)


def apply(patches: list[Patch]) -> list[Path]:
    by_file: dict[Path, list[Patch]] = {}
    for p in patches:
        by_file.setdefault(p.file, []).append(p)
    for f, ps in by_file.items():
        new = apply_edits(f.read_text(encoding="utf-8"), [e for p in ps for e in p.edits])
        ast.parse(new)  # never write a file that no longer parses
        f.write_text(new, encoding="utf-8")
    return list(by_file)
