"""Python signatures / docstrings / pydantic Field() → ParamSchema, statically."""

from __future__ import annotations

import ast
import re
from typing import Any

from agenticreq.models.observed import ParamSchema

_PY_TO_JSON = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "List": "array",
    "tuple": "array",
    "Tuple": "array",
    "set": "array",
    "Sequence": "array",
    "dict": "object",
    "Dict": "object",
    "Mapping": "object",
    "Any": None,
    "bytes": "string",
    "date": "string",
    "datetime": "string",
    "Decimal": "number",
    "UUID": "string",
}
_FIELD_CONSTRAINTS = {
    "le": "maximum",
    "ge": "minimum",
    "lt": "exclusiveMaximum",
    "gt": "exclusiveMinimum",
    "max_length": "maxLength",
    "min_length": "minLength",
    "pattern": "pattern",
}
#: parameters frameworks inject and never show to the model
INJECTED_PARAMS = {
    "self",
    "cls",
    "ctx",
    "context",
    "run_context",
    "config",
    "callbacks",
    "run_manager",
    "tool_call_id",
    "state",
    "store",
    "runtime",
}


#: annotations that mark a parameter the framework fills in (never shown to the model)
_INJECTED_ANNOTATIONS = (
    "RunContext",
    "ToolContext",
    "InjectedState",
    "InjectedToolArg",
    "InjectedStore",
    "InjectedToolCallId",
    "RunnableConfig",
    "CallbackManager",
    "ToolRuntime",
    "BaseStore",
)


def literal(node: ast.AST | None) -> tuple[bool, Any]:
    if node is None:
        return False, None
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False, None


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _name(node.value)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def annotation_info(node: ast.AST | None) -> tuple[str | None, bool, dict[str, Any], str | None]:
    """→ (json_type, is_optional, constraints, description) from an annotation."""
    if node is None:
        return None, False, {}, None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):  # string annotation
        try:
            return annotation_info(ast.parse(node.value, mode="eval").body)
        except SyntaxError:
            return None, False, {}, None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):  # X | None
        parts = [node.left, node.right]
        non_none = [p for p in parts if not (isinstance(p, ast.Constant) and p.value is None)]
        opt = len(non_none) < len(parts)
        if len(non_none) == 1:
            t, o2, c, d = annotation_info(non_none[0])
            return t, opt or o2, c, d
        return None, opt, {}, None
    if isinstance(node, ast.Subscript):
        base = _name(node.value)
        sl = node.slice
        args = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
        if base == "Optional":
            t, _, c, d = annotation_info(args[0])
            return t, True, c, d
        if base == "Union":
            non_none = [a for a in args if not (isinstance(a, ast.Constant) and a.value is None) and _name(a) != "None"]
            t, _, c, d = annotation_info(non_none[0]) if len(non_none) == 1 else (None, False, {}, None)
            return t, len(non_none) < len(args), c, d
        if base == "Annotated":
            t, opt, c, d = annotation_info(args[0])
            for meta in args[1:]:
                if isinstance(meta, ast.Constant) and isinstance(meta.value, str):
                    d = d or meta.value
                elif isinstance(meta, ast.Call) and _name(meta.func) in ("Field", "Doc"):
                    fc, fd = field_call_info(meta)
                    c = {**c, **fc}
                    d = fd or d
            return t, opt, c, d
        if base == "Literal":
            vals = [literal(a)[1] for a in args]
            t = (
                "string"
                if all(isinstance(v, str) for v in vals)
                else ("integer" if all(isinstance(v, int) for v in vals) else None)
            )
            return t, False, {"enum": vals}, None
        return _PY_TO_JSON.get(base, "object"), False, {}, None
    n = _name(node)
    if n == "None":
        return None, True, {}, None
    return _PY_TO_JSON.get(n, "object" if n and n[0].isupper() else None), False, {}, None


def field_call_info(call: ast.Call) -> tuple[dict[str, Any], str | None]:
    c: dict[str, Any] = {}
    desc = None
    for kw in call.keywords:
        if kw.arg == "description":
            ok, v = literal(kw.value)
            desc = v if ok and isinstance(v, str) else desc
        elif kw.arg in _FIELD_CONSTRAINTS:
            ok, v = literal(kw.value)
            if ok:
                c[_FIELD_CONSTRAINTS[kw.arg]] = v
    return c, desc


def field_default(call: ast.Call) -> tuple[bool, Any]:
    """Default from Field(default, ...) / Field(default=...). Field(...) (Ellipsis) means required."""
    node: ast.expr | None
    if call.args:
        node = call.args[0]
    else:
        node = next((k.value for k in call.keywords if k.arg == "default"), None)
    if node is None:
        if any(k.arg == "default_factory" for k in call.keywords):
            return True, None
        return False, None
    ok, v = literal(node)
    if ok and v is Ellipsis:
        return False, None
    return True, v if ok else ast.unparse(node)


# ---------------------------------------------------------------- docstrings
_SECTION = re.compile(r"^\s*(Args|Arguments|Parameters|Params|Returns|Return|Raises|Yields|Examples?|Notes?)\s*:\s*$")


def parse_docstring(doc: str | None) -> tuple[str | None, dict[str, str]]:
    """→ (summary, {param: description}). Google, Sphinx (:param x:) and numpy styles."""
    if not doc:
        return None, {}
    lines = doc.expandtabs().splitlines()
    params: dict[str, str] = {}
    summary_lines: list[str] = []
    section: str | None = None
    current: str | None = None
    for raw in lines:
        line = raw.rstrip()
        m = _SECTION.match(line)
        if m:
            section = m.group(1).lower()
            current = None
            continue
        sp = re.match(r"^\s*:param\s+(?:[\w\[\], ]+\s+)?(\w+)\s*:\s*(.*)$", line)
        if sp:
            params[sp.group(1)] = sp.group(2).strip()
            current = sp.group(1)
            section = "sphinx"
            continue
        if section in ("args", "arguments", "parameters", "params"):
            gm = re.match(r"^\s*(\*{0,2}\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)$", line)
            if gm and not line.strip().startswith("-"):
                current = gm.group(1).lstrip("*")
                params[current] = gm.group(3).strip()
            elif re.match(r"^\s*(\w+)\s*$", line) and section == "parameters":  # numpy "name : type" handled above
                current = line.strip()
                params[current] = ""
            elif current and line.strip() and not re.match(r"^\s*-{3,}\s*$", line):
                params[current] = (params[current] + " " + line.strip()).strip()
            continue
        if section is None:
            summary_lines.append(line)
        elif section == "sphinx" and current and line.strip() and not line.strip().startswith(":"):
            params[current] = (params[current] + " " + line.strip()).strip()
    summary = "\n".join(summary_lines).strip() or None
    return summary, {k: v for k, v in params.items() if v}


def params_from_function(
    fn: ast.FunctionDef | ast.AsyncFunctionDef, doc_params: dict[str, str], skip_first_ctx: bool = False
) -> list[ParamSchema]:
    a = fn.args
    positional = list(a.posonlyargs) + list(a.args)
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
    pairs = list(zip(positional, defaults)) + list(zip(a.kwonlyargs, a.kw_defaults))
    out = []
    for i, (arg, default) in enumerate(pairs):
        if skip_first_ctx and i == 0 or arg.arg in ("self", "cls"):
            continue
        ann = ast.unparse(arg.annotation) if arg.annotation is not None else ""
        if any(m in ann for m in _INJECTED_ANNOTATIONS):
            continue
        t, opt, cons, desc = annotation_info(arg.annotation)
        # `context: str` is a real argument; `config`, `ctx: RunContext[...]`, `state` are framework-injected
        if arg.arg in INJECTED_PARAMS and t not in ("string", "integer", "number", "boolean", "array"):
            continue
        has_default = default is not None
        dval: Any = None
        if isinstance(default, ast.Call) and _name(default.func) == "Field":
            fc, fd = field_call_info(default)
            cons = {**cons, **fc}
            desc = fd or desc
            has_default, dval = field_default(default)
        elif default is not None:
            ok, v = literal(default)
            dval = v if ok else ast.unparse(default)
        out.append(
            ParamSchema(
                name=arg.arg,
                type=t,
                description=desc or doc_params.get(arg.arg),
                required=not has_default,
                has_default=has_default,
                default=dval,
                constraints=cons,
            )
        )
        del opt
    return out


def params_from_model_class(cls: ast.ClassDef) -> list[ParamSchema]:
    """Fields of a pydantic BaseModel (CrewAI args_schema, etc.)."""
    out = []
    for stmt in cls.body:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        name = stmt.target.id
        if name.startswith("_") or name == "model_config":
            continue
        t, opt, cons, desc = annotation_info(stmt.annotation)
        has_default = stmt.value is not None
        dval: Any = None
        if isinstance(stmt.value, ast.Call) and _name(stmt.value.func) == "Field":
            fc, fd = field_call_info(stmt.value)
            cons, desc = {**cons, **fc}, fd or desc
            has_default, dval = field_default(stmt.value)
        elif stmt.value is not None:
            ok, v = literal(stmt.value)
            dval = v if ok else ast.unparse(stmt.value)
        out.append(
            ParamSchema(
                name=name,
                type=t,
                description=desc,
                required=not has_default,
                has_default=has_default,
                default=dval,
                constraints=cons,
            )
        )
        del opt
    return out


def params_from_smolagents_inputs(node: ast.AST) -> list[ParamSchema]:
    ok, inputs = literal(node)
    if not ok or not isinstance(inputs, dict):
        return []
    out = []
    for name, spec in inputs.items():
        spec = spec if isinstance(spec, dict) else {}
        nullable = bool(spec.get("nullable"))
        out.append(
            ParamSchema(
                name=str(name),
                type=spec.get("type"),
                description=spec.get("description"),
                required=not nullable,
                has_default=False,
            )
        )
    return out
