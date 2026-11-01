"""Does a gate call run on every path to a tool's action? Intraprocedural, syntactic dominance.

A gate *covers* a tool body (``all_paths``) only when:

1. the call sits in a top-level statement of the function body and nothing between it and that statement
   makes it conditional (an ``if`` branch, a loop body, a later ``and``/``or`` operand, a ``try`` whose
   handler can carry on, a nested function); the test of a top-level ``if`` is fine, it always runs;
2. its decision is not thrown away (a bare ``interrupt(...)`` statement resumes with whatever the
   human answered);
3. nothing the body does after the gate is also done before it: an early branch that already calls the
   protected operation (``if urgent: self._post(p); return``) is a bypass.

Anything less is ``partial`` (the gate is on some paths only) or ``unverified``. Never a PASS.
"""

from __future__ import annotations

import ast
from typing import Literal

Coverage = Literal["all_paths", "partial", "unverified"]

#: calls that read or build values; seeing one both before and after the gate is not a bypass
_PURE = {
    "str",
    "int",
    "float",
    "bool",
    "dict",
    "list",
    "tuple",
    "set",
    "len",
    "isinstance",
    "min",
    "max",
    "sorted",
    "any",
    "all",
    "round",
    "abs",
    "sum",
    "range",
    "enumerate",
    "zip",
    "print",
    "repr",
    "format",
    "get",
    "items",
    "keys",
    "values",
    "copy",
    "strip",
    "lower",
    "upper",
    "join",
    "split",
    "startswith",
    "endswith",
    "replace",
    "model_dump",
    "model_validate",
    "dumps",
    "loads",
    "time",
    "now",
    "isoformat",
    "hexdigest",
    "getLogger",
    "info",
    "debug",
    "warning",
    "error",
    "exception",
    "log",
}
#: verbs of calls that change the outside world: one of these BEFORE the gate means the tool already acted
_ACTING = (
    "post",
    "put",
    "patch",
    "delete",
    "send",
    "write",
    "pay",
    "charge",
    "transfer",
    "refund",
    "submit",
    "create",
    "insert",
    "update",
    "remove",
    "drop",
    "execute",
    "publish",
    "commit",
    "save",
    "book",
    "cancel",
    "approve",
    "issue",
    "withdraw",
    "deposit",
    "wire",
    "email",
    "notify",
    "upload",
    "run",
    "exec",
    "system",
    "call",
)
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _effects(stmts: list[ast.stmt], skip: ast.AST | None = None) -> set[str]:
    """Names of the operations these statements perform: non-pure calls and writes into shared state."""
    out: set[str] = set()
    skipped = {id(n) for n in ast.walk(skip)} if skip is not None else set()
    for st in stmts:
        for n in ast.walk(st):
            if id(n) in skipped:
                continue
            if isinstance(n, ast.Call):
                name = _call_name(n.func)
                if name and name not in _PURE and not name[:1].isupper():  # Model(...) builds a value
                    out.add(f"call:{name}")
            elif isinstance(n, (ast.Subscript, ast.Attribute)) and isinstance(n.ctx, ast.Store):
                out.add(f"store:{ast.unparse(n.value)}")  # REFUNDS[key] = ..., self.state = ...
    return out


def gate_coverage(fn: ast.FunctionDef | ast.AsyncFunctionDef, call: ast.Call) -> tuple[Coverage, str]:
    """(coverage, why) for one gate call inside one tool function / method."""
    parents: dict[int, ast.AST] = {}
    for p in ast.walk(fn):
        for c in ast.iter_child_nodes(p):
            parents[id(c)] = p
    line = getattr(call, "lineno", "?")
    chain: list[tuple[ast.AST, ast.AST]] = []
    node: ast.AST = call
    while id(node) in parents and parents[id(node)] is not fn:
        parent = parents[id(node)]
        chain.append((parent, node))
        node = parent
    if not any(st is node for st in fn.body):
        return "unverified", f"gate at line {line} is not in the body of {fn.name}"
    top = node
    for parent, child in chain:
        if isinstance(parent, _FUNCS) or isinstance(parent, _COMPS):
            return "unverified", f"gate at line {line} is inside a nested function / comprehension"
        if isinstance(parent, ast.If) and child is not parent.test:
            return (
                "partial",
                f"gate at line {line} runs only inside the `if {ast.unparse(parent.test)}` branch "
                f"(line {parent.lineno}); the other branch reaches the tool's action without it",
            )
        if isinstance(parent, ast.IfExp) and child is not parent.test:
            return "partial", f"gate at line {line} is one arm of a conditional expression"
        if isinstance(parent, ast.BoolOp) and child is not parent.values[0]:
            return (
                "partial",
                f"gate at line {line} is a later operand of `{type(parent.op).__name__.lower()}`; "
                "short-circuiting can skip it",
            )
        if isinstance(parent, (ast.For, ast.AsyncFor)) and child is not parent.iter:
            return (
                "partial",
                f"gate at line {line} is inside a loop body (line {parent.lineno}) that may run zero times",
            )
        if isinstance(parent, ast.While) and child is not parent.test:
            return "partial", f"gate at line {line} is inside a while body (line {parent.lineno})"
        if isinstance(parent, ast.Match):
            return "partial", f"gate at line {line} is inside one `match` case"
        if isinstance(parent, ast.Try) or type(parent).__name__ == "TryStar":
            return (
                "unverified",
                f"gate at line {line} is inside `try` (line {getattr(parent, 'lineno', '?')}); "
                "a handler can continue without approval",
            )
    if isinstance(top, ast.Expr):
        value = top.value.value if isinstance(top.value, ast.Await) else top.value
        if isinstance(value, ast.Call) and _call_name(value.func) == "interrupt":
            return "unverified", f"the decision returned by interrupt() at line {line} is discarded"
    idx = next(i for i, st in enumerate(fn.body) if st is top)
    after = fn.body[idx + 1 :]
    protected = _effects(after) | (_effects(top.body + top.orelse) if isinstance(top, ast.If) else set())
    before = _effects(fn.body[:idx])
    acted_first = sorted(e for e in before if e.startswith("call:") and e[5:].lower().lstrip("_").startswith(_ACTING))
    if acted_first:  # the action itself ran before the human was asked
        what = ", ".join(e.split(":", 1)[1] for e in acted_first[:3])
        return (
            "partial",
            f"{what} runs before the gate at line {line}, so the tool has already acted when approval is asked",
        )
    bypass = sorted(protected & before)
    if bypass:
        what = ", ".join(b.split(":", 1)[1] for b in bypass[:3])
        return "partial", f"{what} is reachable before the gate at line {line}, so a path can act without it"
    return "all_paths", f"gate at line {line} runs on every path through {fn.name} before its action"
