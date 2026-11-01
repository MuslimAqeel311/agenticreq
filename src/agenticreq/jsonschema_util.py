"""JSON-schema (tool parameters) → ParamSchema. Shared by capture, MCP and traces."""

from __future__ import annotations

from typing import Any

from agenticreq.models.observed import ParamSchema, ToolSchema, ToolSource

_CONSTRAINT_KEYS = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "maxLength",
    "minLength",
    "maxItems",
    "minItems",
    "enum",
    "pattern",
)


def _type_of(prop: dict[str, Any]) -> str | None:
    t = prop.get("type")
    if isinstance(t, str):
        return t
    if isinstance(t, list):
        non_null = [str(x) for x in t if x != "null"]
        return non_null[0] if len(non_null) == 1 else "|".join(non_null) or None
    for key in ("anyOf", "oneOf"):
        alts = prop.get(key)
        if isinstance(alts, list):
            types = [
                str(a["type"])
                for a in alts
                if isinstance(a, dict) and isinstance(a.get("type"), str) and a.get("type") != "null"
            ]
            if len(types) == 1:
                return types[0]
            if types:
                return "|".join(types)
    if "enum" in prop:
        return "string"
    if "$ref" in prop:
        return "object"
    return None


def _constraints(prop: dict[str, Any]) -> dict[str, Any]:
    out = {k: prop[k] for k in _CONSTRAINT_KEYS if k in prop}
    for key in ("anyOf", "oneOf"):
        for alt in prop.get(key) or []:
            if isinstance(alt, dict):
                out.update({k: alt[k] for k in _CONSTRAINT_KEYS if k in alt and k not in out})
    return out


def params_from_json_schema(schema: dict[str, Any] | None) -> list[ParamSchema]:
    if not isinstance(schema, dict):
        return []
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    out = []
    for name, prop in props.items():
        if not isinstance(prop, dict):
            prop = {}
        out.append(
            ParamSchema(
                name=name,
                type=_type_of(prop),
                description=prop.get("description") or None,
                required=name in required,
                has_default="default" in prop,
                default=prop.get("default"),
                constraints=_constraints(prop),
            )
        )
    return out


def tool_from_json_schema(
    name: str, description: str | None, parameters: dict[str, Any] | None, source: ToolSource, kind: str
) -> ToolSchema:
    return ToolSchema(
        name=name,
        description=description or None,
        params=params_from_json_schema(parameters),
        source=source,
        kind=kind,
        raw={"name": name, "description": description, "parameters": parameters},
    )


def example_value(prop: dict[str, Any]) -> Any:
    """Deterministic placeholder argument for a JSON-schema property (canned responder)."""
    if "default" in prop:
        return prop["default"]
    if prop.get("enum"):
        return prop["enum"][0]
    t = _type_of(prop)
    if t == "integer":
        lo = prop.get("minimum", 1)
        return int(lo) if isinstance(lo, (int, float)) else 1
    if t == "number":
        lo = prop.get("minimum", 1.0)
        return float(lo) if isinstance(lo, (int, float)) else 1.0
    if t == "boolean":
        return False
    if t == "array":  # one item, so a tool that loops over it does something
        item = prop.get("items")
        return [example_value(item)] if isinstance(item, dict) and item else ["example"]
    if t == "object":
        return {}
    return "example"


def named_example(name: str, prop: dict[str, Any]) -> Any:
    """``example_value`` with a string that fits the parameter's name (a url, a path, a query, an id)."""
    v = example_value(prop)
    if v != "example":
        return v
    n = name.lower()
    for words, text in (
        (("url", "link", "href"), "https://example.com"),
        (("path", "file", "filename"), ".agenticreq/mock_run/example.txt"),
        (("email",), "user@example.com"),
        (("id",), "1"),
        (("date",), "2026-01-01"),
        (("ticker", "symbol"), "AAPL"),
    ):
        if any(w == n or n.endswith("_" + w) or n.startswith(w + "_") for w in words):
            return text
    return f"example {name.replace('_', ' ')}"


def example_arguments(schema: dict[str, Any] | None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    args: dict[str, Any] = {}
    if isinstance(schema, dict):
        props = schema.get("properties") or {}
        for name in schema.get("required") or []:
            args[name] = named_example(name, props.get(name) or {})
    args.update(overrides or {})
    return args


def example_instance(
    schema: dict[str, Any] | None,
    overrides: dict[str, Any] | None = None,
    _root: dict[str, Any] | None = None,
    _depth: int = 0,
) -> Any:
    """A deterministic value that validates against ``schema`` (structured-output answers).

    Follows ``$ref`` into ``$defs`` / ``definitions``, takes the first non-null branch of
    ``anyOf`` / ``oneOf``, and fills **every** property (strict JSON-schema mode requires all).
    Booleans default to ``True`` so "is_relevant" / "is_safe" style guardrails let the run continue;
    ``overrides`` (from mocks.yml ``structured:``) set any field by name, at any depth.
    """
    root = _root if _root is not None else (schema or {})
    if not isinstance(schema, dict) or _depth > 12:
        return None
    if "$ref" in schema:
        ref = str(schema["$ref"])
        target: Any = root
        for part in ref.lstrip("#/").split("/"):
            target = target.get(part, {}) if isinstance(target, dict) else {}
        return example_instance(target, overrides, root, _depth + 1)
    for key in ("anyOf", "oneOf"):
        if isinstance(schema.get(key), list):
            branches = [b for b in schema[key] if isinstance(b, dict) and b.get("type") != "null"]
            return example_instance(branches[0] if branches else {}, overrides, root, _depth + 1)
    if isinstance(schema.get("allOf"), list) and schema["allOf"]:
        return example_instance(schema["allOf"][0], overrides, root, _depth + 1)
    if "const" in schema:
        return schema["const"]
    t = _type_of(schema)
    if t == "object" or "properties" in schema:
        out: dict[str, Any] = {}
        for name, prop in (schema.get("properties") or {}).items():
            out[name] = (overrides or {}).get(name, example_instance(prop, overrides, root, _depth + 1))
        return out
    if t == "array":
        n = int(schema.get("minItems", 0) or 0)
        return [example_instance(schema.get("items") or {}, overrides, root, _depth + 1) for _ in range(n)]
    if t == "boolean" and "default" not in schema:
        return True
    return example_value(schema)
