"""A wrong field name in spec.yml: stop, suggest the closest real one, never continue silently, never guess when two
are equally close. Plus `validate --fix` renames, the JSON Schema for editors, and `params` showing "also called"."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.loader import load_spec_text
from agenticreq.loader.issues import normalise, suggest
from agenticreq.schema import MODELINE, spec_json_schema

runner = CliRunner()
FIX = Path(__file__).parent / "fixtures"
PASSING = FIX / "passfail" / "captured"
BASE = (PASSING / "spec.yml").read_text(encoding="utf-8")


# ================================================================== the suggestion engine
@pytest.mark.parametrize(
    "name,parent,expected",
    [
        ("prohibted", "", ["prohibited"]),  # a typo
        ("maxSteps", "bounds", ["bounds.max_steps"]),  # camelCase
        ("max-steps", "bounds", ["bounds.max_steps"]),  # kebab-case
        ("forbidden_tools", "", ["prohibited"]),  # an "also called" word of the catalog entry
        ("max_iterations", "", ["bounds.max_steps"]),  # a nested field, from the top level, via "also called"
        ("owners", "", ["owner"]),  # a plural
        ("human_approval", "", ["approval_required"]),
        ("timout", "approval_required[]", ["approval_required[].timeout"]),  # inside a list entry
        ("approvals_required", "", ["approval_required"]),
        ("prohibted_tools", "", ["prohibited"]),  # not role_boundary ("permitted_tools"): shared words do not count
    ],
)
def test_closest_field(name: str, parent: str, expected: list[str]) -> None:
    assert suggest(name, parent) == expected


def test_two_equally_close_fields_are_both_listed() -> None:
    assert suggest("timeout") == ["approval_required[].timeout", "bounds.max_duration_s"]
    both = suggest("maxSteps")  # at the top level: two mappings hold a max_steps
    assert "bounds.max_steps" in both and len(both) == 2


def test_no_close_field_gives_no_guess() -> None:
    assert suggest("zzzz") == [] and suggest("colour_scheme") == []


def test_normalise() -> None:
    assert normalise("maxSteps") == normalise("max-steps") == normalise("Max_Steps") == "max_step"
    assert normalise("the_agent_owner") == "owner"


# ================================================================== check stops, with one plain message
def _spec_with(line: str) -> str:
    head, tail = BASE.split("confirmed:", 1)
    return head + line + "\n" + "confirmed:" + tail


def _issue(text: str):  # type: ignore[no-untyped-def]
    loaded = load_spec_text(text, "spec.yml")
    return loaded, next(i for i in loaded.issues if i.rule_id == "AS-117")


def test_unknown_field_message_with_a_suggestion() -> None:
    text = _spec_with("max_iterations: 12")
    loaded, issue = _issue(text)
    line = text.splitlines().index("max_iterations: 12") + 1
    assert loaded.status == "invalid"
    assert issue.message == (
        f"Unknown field `max_iterations` in spec.yml (line {line}). Did you mean "
        "`bounds.max_steps`? Fix it, or run: agenticreq validate --fix"
    )


def test_unknown_field_message_with_two_and_with_none() -> None:
    _, two = _issue(_spec_with("timeout: 4h"))
    assert "Did you mean `approval_required[].timeout` or `bounds.max_duration_s`?" in two.message
    _, none = _issue(_spec_with("colour_scheme: dark"))
    assert none.message.startswith("Unknown field `colour_scheme` in spec.yml (line ")
    assert "Did you mean" not in none.message and none.did_you_mean == []


def test_check_exits_2_on_an_unknown_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = tmp_path / "app"
    shutil.copytree(PASSING, d)
    (d / "spec.yml").write_text(_spec_with("prohibted: []"), encoding="utf-8")
    monkeypatch.chdir(d)
    r = runner.invoke(app, ["check", "--quiet"])
    out = " ".join(r.output.split())
    assert r.exit_code == 2
    assert "Unknown field `prohibted` in spec.yml" in out and "Did you mean `prohibited`?" in out
    assert "unknown field" in out  # the issue's label in the SPEC INVALID list


# ================================================================== validate --fix
def _fix(tmp_path: Path, text: str, *args: str, answer: str | None = None) -> tuple[int, str, str]:
    p = tmp_path / "spec.yml"
    p.write_text(text, encoding="utf-8")
    r = runner.invoke(app, ["validate", str(p), "--fix", *args], input=answer)
    return r.exit_code, " ".join(r.output.split()), p.read_text(encoding="utf-8")


def test_fix_applies_one_clear_rename_and_keeps_comments(tmp_path: Path) -> None:
    text = BASE.replace(
        "progress_guard: {max_no_progress_steps: 5}",
        "progress_guard: {max_no_progress_steps: 5}\nbounds: {maxSteps: 10}   # our cap",
    )
    code, out, after = _fix(tmp_path, text, "--yes")
    assert "rename `bounds.maxSteps` -> `bounds.max_steps`" in out and "wrote" in out
    assert "bounds: {max_steps: 10}   # our cap" in after  # only the key changed; the comment stays
    assert [ln for ln in after.splitlines() if ln.startswith("#")] == [
        ln for ln in text.splitlines() if ln.startswith("#")
    ]
    assert load_spec_text(after).status != "invalid" and code == 0


def test_fix_never_renames_an_ambiguous_field(tmp_path: Path) -> None:
    text = _spec_with("timeout: 4h")
    code, out, after = _fix(tmp_path, text, "--yes")
    assert "not renamed `timeout`" in out and "equally close" in out
    assert after == text and code == 2


def test_fix_asks_before_writing(tmp_path: Path) -> None:
    text = _spec_with("prohibted_tools: []").replace("prohibited: []\n", "")
    code, out, after = _fix(tmp_path, text, answer="n\n")
    assert "Write these changes" in out and "nothing written" in out and after == text
    code, out, after = _fix(tmp_path, text, answer="y\n")
    assert "rename `prohibted_tools` -> `prohibited`" in out and "prohibited: []" in after


def test_fix_does_not_rename_onto_a_field_that_is_already_there(tmp_path: Path) -> None:
    code, out, after = _fix(tmp_path, _spec_with("prohibted: []"), "--yes")  # prohibited: [] is already set
    assert "already there" in out and "prohibted: []" in after


def test_fix_leaves_a_field_that_belongs_in_another_mapping(tmp_path: Path) -> None:
    code, out, after = _fix(tmp_path, _spec_with("max_iterations: 12"), "--yes")
    assert "move it to `bounds.max_steps` by hand" in out and "max_iterations: 12" in after


# ================================================================== the JSON Schema
def _yaml12(text: str) -> Any:
    """Load as an editor does (YAML 1.2): a date stays text."""

    class Loader(yaml.SafeLoader):
        pass

    Loader.yaml_implicit_resolvers = {
        k: [r for r in v if r[0] != "tag:yaml.org,2002:timestamp"]
        for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    return yaml.load(text, Loader=Loader)  # noqa: S506 - a SafeLoader subclass


def _unknown_keys(node: Any, schema: dict[str, Any], defs: dict[str, Any], path: str = "") -> list[str]:
    """Keys the schema does not allow (resolves $ref, anyOf, properties, additionalProperties, items)."""
    if "$ref" in schema:
        return _unknown_keys(node, defs[schema["$ref"].rsplit("/", 1)[-1]], defs, path)
    if "anyOf" in schema:
        tries = [_unknown_keys(node, alt, defs, path) for alt in schema["anyOf"] if _fits(node, alt, defs)]
        return min(tries, key=len) if tries else [path or "(root)"]
    out: list[str] = []
    if isinstance(node, dict) and schema.get("type") in (None, "object"):
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for k, v in node.items():
            sub = f"{path}.{k}" if path else str(k)
            if k in props:
                out += _unknown_keys(v, props[k], defs, sub)
            elif extra is False:
                out.append(sub)
            elif isinstance(extra, dict):
                out += _unknown_keys(v, extra, defs, sub)
    elif isinstance(node, list) and isinstance(schema.get("items"), dict):
        for i, v in enumerate(node):
            out += _unknown_keys(v, schema["items"], defs, f"{path}[{i}]")
    return out


def _fits(node: Any, schema: dict[str, Any], defs: dict[str, Any]) -> bool:
    if "$ref" in schema:
        return _fits(node, defs[schema["$ref"].rsplit("/", 1)[-1]], defs)
    t = schema.get("type")
    if "const" in schema:
        return bool(node == schema["const"])
    return {
        "object": isinstance(node, dict),
        "array": isinstance(node, list),
        "string": isinstance(node, str),
        "null": node is None,
        "boolean": isinstance(node, bool),
        "integer": isinstance(node, int) and not isinstance(node, bool),
        "number": isinstance(node, (int, float)) and not isinstance(node, bool),
    }.get(str(t), True)


def test_schema_accepts_a_correct_spec_and_rejects_an_unknown_key() -> None:
    schema = spec_json_schema()
    defs = schema.get("$defs", {})
    good = _yaml12(BASE)
    assert _unknown_keys(good, schema, defs) == []
    assert _unknown_keys({**good, "max_iterations": 3}, schema, defs) == ["max_iterations"]
    assert _unknown_keys({**good, "bounds": {"maxSteps": 3}}, schema, defs) == ["bounds.maxSteps"]


def test_schema_with_a_full_validator() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    v = jsonschema.Draft202012Validator(spec_json_schema())
    good = _yaml12(BASE)
    assert list(v.iter_errors(good)) == []
    assert list(v.iter_errors({**good, "max_iterations": 3}))
    assert list(v.iter_errors(_yaml12(BASE.replace("owner: owner@example.com", "owner: TODO")))) == []


def test_schema_describes_fields_from_the_catalog() -> None:
    props = spec_json_schema()["properties"]
    assert "forbidden" in json.dumps(props["prohibited"]) and "max_iterations" in json.dumps(props["bounds"])


def test_export_schema_and_init_modeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = runner.invoke(app, ["export-schema"])
    assert r.exit_code == 0 and json.loads(r.stdout)["title"] == "agenticreq spec.yml"
    shutil.copy(PASSING / "agent.py", tmp_path / "agent.py")
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "--quiet"]).exit_code == 0
    assert (tmp_path / "spec.yml").read_text(encoding="utf-8").splitlines()[0] == MODELINE
    assert json.loads((tmp_path / "spec.schema.json").read_text(encoding="utf-8"))["title"] == "agenticreq spec.yml"


# ================================================================== params shows "also called"
def test_params_shows_also_called() -> None:
    out = runner.invoke(app, ["params", "--field", "prohibited"]).output
    assert "also called: forbidden, banned, blocked" in out
    data = json.loads(runner.invoke(app, ["params", "--field", "bounds", "-f", "json"]).stdout)
    assert data[0]["also_called"]["bounds.max_steps"][:3] == ["max_iterations", "max_turns", "recursion_limit"]
