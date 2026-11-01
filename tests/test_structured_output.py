"""Canned responder answers structured-output requests with schema-valid JSON (validation H1)."""

import json

from agenticreq.capture.responder import CannedResponder, MockConfig, output_schema
from agenticreq.capture.wire import parse_request
from agenticreq.jsonschema_util import example_instance

GUARDRAIL = {  # shape of an OpenAI Agents output_type schema (pydantic, strict)
    "type": "object",
    "additionalProperties": False,
    "required": ["reasoning", "is_relevant"],
    "properties": {"reasoning": {"type": "string"}, "is_relevant": {"type": "boolean"}},
}
PLAN = {
    "$defs": {
        "Item": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "kind": {"enum": ["web", "file"]}},
            "required": ["query", "kind"],
        }
    },
    "type": "object",
    "required": ["searches"],
    "properties": {
        "searches": {"type": "array", "items": {"$ref": "#/$defs/Item"}, "minItems": 1},
        "note": {"anyOf": [{"type": "null"}, {"type": "string"}]},
    },
}


def test_example_instance_follows_refs_and_fills_all_properties() -> None:
    assert example_instance(PLAN) == {"searches": [{"query": "example", "kind": "web"}], "note": "example"}
    assert example_instance(GUARDRAIL) == {"reasoning": "example", "is_relevant": True}
    assert example_instance(GUARDRAIL, {"is_relevant": False})["is_relevant"] is False


def test_output_schema_per_provider() -> None:
    assert output_schema({"text": {"format": {"type": "json_schema", "schema": PLAN}}}, "openai_responses") == PLAN
    assert (
        output_schema({"response_format": {"type": "json_schema", "json_schema": {"schema": PLAN}}}, "openai_chat")
        == PLAN
    )
    assert output_schema({"response_format": {"type": "json_object"}}, "openai_chat") == {}
    assert output_schema({"messages": []}, "openai_chat") is None


def test_responses_api_final_answer_is_json() -> None:
    body = {
        "model": "gpt-4.1",
        "input": [{"role": "user", "content": "hi"}],
        "text": {"format": {"type": "json_schema", "name": "out", "schema": GUARDRAIL, "strict": True}},
    }
    r = CannedResponder(MockConfig())
    call = parse_request(0, "https://api.openai.com/v1/responses", body)
    raw, _ = r.render(call, r.plan(call, body), stream=False)
    text = json.loads(raw)["output"][0]["content"][0]["text"]
    assert json.loads(text) == {"reasoning": "example", "is_relevant": True}


def test_plain_requests_still_get_prose() -> None:
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    r = CannedResponder(MockConfig(final_text="Done."))
    call = parse_request(0, "https://api.openai.com/v1/chat/completions", body)
    raw, _ = r.render(call, r.plan(call, body), stream=False)
    assert json.loads(raw)["choices"][0]["message"]["content"] == "Done."
