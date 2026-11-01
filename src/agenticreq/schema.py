"""JSON Schema for spec.yml (`agenticreq export-schema > spec.schema.json`), so an editor completes field names and
flags a wrong one as it is typed.

Built from the spec model (every field and nested field, unknown names rejected at every level) and the params
catalog (each field's level, group, note and "also called" names in its description). It also accepts what the
loader accepts: the short forms (``prohibited: [tool_name]``, ``limits: {tool: {arg: {max: 5}}}``, ``goal: "..."``)
and a ``TODO`` left in any field, which `check` reports as a gap rather than the editor as an error.
"""

from __future__ import annotations

import copy
from typing import Any

SCHEMA_FILE = "spec.schema.json"
#: the first line `agenticreq init` writes, read by the YAML extension of VS Code (and other editors)
MODELINE = f"# yaml-language-server: $schema=./{SCHEMA_FILE}"
_TODO: dict[str, Any] = {"const": "TODO", "description": "not written yet: `agenticreq check` reports it as a gap"}


#: any field may be written as not_needed (not_needed.py)
_NOT_NEEDED: dict[str, Any] = {"const": "not_needed", "description": "this does not apply to the project"}


def _or_todo(node: Any) -> Any:
    """Every property may also be TODO, or not_needed."""
    if isinstance(node, dict):
        out = {k: _or_todo(v) for k, v in node.items()}
        props = out.get("properties")
        if isinstance(props, dict):
            out["properties"] = {k: {"anyOf": [v, _TODO, _NOT_NEEDED]} for k, v in props.items()}
        return out
    if isinstance(node, list):
        return [_or_todo(x) for x in node]
    return node


def _named_list(item_ref: dict[str, Any]) -> dict[str, Any]:
    """prohibited / approval_required / untrusted_inputs: a list of entries or names, or a {name: entry} mapping."""
    return {
        "anyOf": [
            {"type": "array", "items": {"anyOf": [item_ref, {"type": "string"}]}},
            {"type": "object", "additionalProperties": {"anyOf": [{"type": "object"}, {"type": "null"}]}},
        ]
    }


def spec_json_schema() -> dict[str, Any]:
    from agenticreq.models.spec import AgentSpec
    from agenticreq.params import BY_NAME

    schema = AgentSpec.model_json_schema(by_alias=True)
    props: dict[str, Any] = schema["properties"]

    def items_ref(name: str) -> dict[str, Any]:
        node = props[name]
        for alt in node.get("anyOf", [node]):
            if alt.get("type") == "array":
                return copy.deepcopy(alt["items"])
        return {"type": "object"}

    for name in ("prohibited", "approval_required", "untrusted_inputs"):
        props[name] = {**_named_list(items_ref(name)), "default": None}
    limit_item = items_ref("limits")
    props["limits"] = {
        "anyOf": [
            {"type": "array", "items": {"anyOf": [limit_item, {"type": "object"}]}},
            {"type": "object", "additionalProperties": {"type": "object"}},
            {"type": "null"},
        ]
    }
    props["goal"] = {"anyOf": [props["goal"], {"type": "string"}]}
    if "declared_delegation" in props:
        props["declared_delegation"] = {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"anyOf": [items_ref("declared_delegation"), {"type": "string", "pattern": "->"}]},
                },
                {"type": "null"},
            ]
        }
    if "retention" in str(schema.get("$defs", {}).get("DataRules", {})):
        dr = schema["$defs"]["DataRules"]["properties"]
        dr["retention"] = {"anyOf": [dr["retention"], {"type": "string"}]}

    for name, prop in props.items():  # what each field is, and its other names, in the editor's hover
        p = BY_NAME.get(name)
        if p is None:
            continue
        text = f"Level {p.level} ({p.group}). " + (f"{p.note}. " if p.note else "")
        also = [f"{full} ({', '.join(words)})" for full, words in p.also_called()]
        prop["description"] = (text + (f"Also called: {'; '.join(also)}." if also else "")).strip()

    confirmed = schema.get("$defs", {}).get("Attestation", {}).get("properties", {})
    if "answer" in confirmed:  # a bare yes / no, as a YAML 1.1 reader sees it
        confirmed["answer"] = {"anyOf": [confirmed["answer"], {"type": "boolean"}], "default": "yes"}

    out: dict[str, Any] = _or_todo(schema)
    out["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    out["title"] = "agenticreq spec.yml"
    return out
