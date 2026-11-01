"""P2: exported policies reject a type-mismatched argument instead of allowing it.

Evaluated with the real engines (the `opa` binary and cedarpy), never by string matching. One spec holds a
limit of every kind the spec allows: max / min on a number, max_length and pattern on a string, enum of
strings and enum of numbers. Each gets an in-range value, an out-of-range value and a value of the wrong type.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest

from agenticreq.export import to_cedar, to_rego
from agenticreq.loader import load_spec_text

ROOT = Path(__file__).resolve().parents[1]
OPA = shutil.which("opa") or (str(ROOT / ".tools" / "opa") if (ROOT / ".tools" / "opa").exists() else None)

SPEC = textwrap.dedent("""
    spec_version: 1
    agent_id: typed
    risk_level: medium
    role_boundary: {allowed_tools: [pay, note, set_mode, set_level, tag, lookup]}
    prohibited: [{tool: drop_db, reason: never}]
    approval_required: [{tool: refund, approver: lead}]
    limits:
      - {tool: pay, arg: amount, limit: {min: 1, max: 1000}}
      - {tool: note, arg: text, limit: {max_length: 10}}
      - {tool: set_mode, arg: mode, limit: {enum: [fast, slow]}}
      - {tool: set_level, arg: level, limit: {enum: [1, 2, 3]}}
      - {tool: tag, arg: code, limit: {pattern: "^[A-Z]{3}$"}}
""")

# (tool, args, allowed) — shared by both engines
CASES: list[tuple[str, dict[str, Any], bool]] = [
    # number: max / min
    ("pay", {"amount": 500}, True),
    ("pay", {"amount": 5000}, False),
    ("pay", {"amount": 0}, False),
    ("pay", {"amount": "500"}, False),  # in range, but a string: type mismatch
    ("pay", {"amount": "5000"}, False),  # the scenario's hole: Cedar allowed this
    ("pay", {"amount": "0"}, False),  # a string under a min: Rego ordering would have allowed it
    ("pay", {"amount": True}, False),
    ("pay", {"amount": None}, False),
    # enum of strings
    ("set_mode", {"mode": "fast"}, True),
    ("set_mode", {"mode": "turbo"}, False),
    ("set_mode", {"mode": 1}, False),
    # enum of numbers
    ("set_level", {"level": 2}, True),
    ("set_level", {"level": 5}, False),
    ("set_level", {"level": "2"}, False),  # was allowed-by-stringification risk; must be the Long 2
    # a tool with no limit is untouched
    ("lookup", {"q": "anything"}, True),
]
# string limits: Rego has count() and regex.match; Cedar has neither (see the NOT ENFORCED note)
OPA_ONLY: list[tuple[str, dict[str, Any], bool]] = [
    ("note", {"text": "short"}, True),
    ("note", {"text": "far too long here"}, False),
    ("note", {"text": 12345}, False),
    ("note", {"text": ["a", "b"]}, False),  # count() works on arrays too: must still be a type mismatch
    ("tag", {"code": "ABC"}, True),
    ("tag", {"code": "abcd"}, False),
    ("tag", {"code": 123}, False),
    ("pay", {"amount": 500.5}, True),  # JSON floats are fine for OPA
    ("pay", {"amount": 1000.5}, False),
]


EXPECTED: dict[str, Any] = {"pay": (int, float), "note": str, "tag": str, "set_mode": str, "set_level": int}


def _spec() -> Any:
    s = load_spec_text(SPEC).spec
    assert s is not None
    return s


@pytest.mark.skipif(OPA is None, reason="opa binary not available")
@pytest.mark.parametrize("tool,args,allowed", CASES + OPA_ONLY)
def test_opa_rejects_type_mismatch(tmp_path: Path, tool: str, args: dict[str, Any], allowed: bool) -> None:
    assert OPA is not None
    pol = tmp_path / "p.rego"
    pol.write_text(to_rego(_spec()))
    subprocess.run([OPA, "check", str(pol)], check=True, capture_output=True)
    (tmp_path / "in.json").write_text(json.dumps({"tool": tool, "args": args}))
    out = subprocess.run(
        [OPA, "eval", "-f", "json", "-d", str(pol), "-i", str(tmp_path / "in.json"), "data.agenticreq.typed"],
        check=True,
        capture_output=True,
        text=True,
    )
    res = json.loads(out.stdout)["result"][0]["expressions"][0]["value"]
    assert res["allow"] is allowed, res["deny"]
    v = next(iter(args.values()))
    if tool in EXPECTED and (isinstance(v, bool) or not isinstance(v, EXPECTED[tool])):
        # a wrong-typed value is denied as a type mismatch (or as not in the enum), never by a cross-type compare
        assert any("has type" in m or "not in" in m for m in res["deny"]), res["deny"]


cedarpy = pytest.importorskip("cedarpy")


@pytest.mark.parametrize("tool,args,allowed", CASES)
def test_cedar_rejects_type_mismatch(tool: str, args: dict[str, Any], allowed: bool) -> None:
    req = {
        "principal": 'Agent::"typed"',
        "action": 'Action::"call_tool"',
        "resource": f'Tool::"{tool}"',
        "context": {"args": args},
    }
    res = cedarpy.is_authorized(req, to_cedar(_spec()), [])
    assert (res.decision == cedarpy.Decision.Allow) is allowed, res.diagnostics


def test_cedar_says_what_it_cannot_enforce() -> None:
    pol = to_cedar(_spec())
    assert "NOT ENFORCED HERE: note.text max_length" in pol
    assert "NOT ENFORCED HERE: tag.code pattern" in pol
