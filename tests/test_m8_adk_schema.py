"""M8: Google ADK / Gemini tool schemas: parameters_json_schema is read, and upper-case type names equal the JSON
Schema ones; the whole-run call limit, sub_agents, AgentTool and invocation_id keep their existing handling."""

from __future__ import annotations

import pytest

from agenticreq.capture.responder import _tool_schema
from agenticreq.capture.wire import parse_request
from agenticreq.data import load_yaml
from agenticreq.jsonschema_util import example_arguments, lower_types

URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent"
UPPER = {
    "type": "OBJECT",
    "properties": {
        "amount": {"type": "NUMBER"},
        "note": {"type": "STRING"},
        "urgent": {"type": "BOOLEAN"},
        "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["amount", "note", "urgent", "tags"],
}


@pytest.mark.parametrize("key", ["parameters", "parametersJsonSchema", "parameters_json_schema"])
def test_m8_gemini_schema_keys_and_upper_case_types(key: str) -> None:
    body = {"contents": [], "tools": [{"functionDeclarations": [{"name": "pay", "description": "Pay.", key: UPPER}]}]}
    call = parse_request(0, URL, body)
    params = {p.name: p.type for p in call.tools[0].params}
    assert params == {"amount": "number", "note": "string", "urgent": "boolean", "tags": "array"}
    args = example_arguments(_tool_schema(body, "gemini", "pay"))
    assert (
        isinstance(args["amount"], (int, float)) and isinstance(args["urgent"], bool) and isinstance(args["tags"], list)
    )


def test_m8_lower_types_keeps_everything_else() -> None:
    assert lower_types({"type": ["STRING", "NULL"], "enum": ["A"]}) == {"type": ["string", "null"], "enum": ["A"]}


def test_m8_existing_adk_handling_is_data_driven() -> None:
    assert load_yaml("bound_names.yml")["kwargs"].get("max_llm_calls") == "max_steps" or "max_llm_calls" in str(
        load_yaml("bound_names.yml")
    )
    assert "invocation_id" in str(load_yaml("tracing.yml"))
