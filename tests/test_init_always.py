"""Point 1: `init` always works and is the default path to a spec. It never needs (or imports) the agent's
dependencies, falls back to AST-only when it cannot run the agent, says in _derived what it could not
resolve, and never writes a spec with a shape error: only TODOs for the user's own decisions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.loader import load_spec
from agenticreq.shapes import example, shape, spec_fields

AGENTS = Path(__file__).parent / "fixtures" / "agents"
runner = CliRunner()

HOSTILE = {
    "unknown_framework": {"bot.py": "import requests\nMODEL = 'gpt-4o'\ndef run(q):\n    return requests.get(q)\n"},
    "empty_dir": {},
    "syntax_error": {"agent.py": "from langchain_core.tools import tool\n@tool\ndef f(:\n"},
    "null_bytes": {"agent.py": "x = 1\x00\n"},
    "latin1_bytes": {"agent.py": b"# caf\xe9\nx = 1\n"},
    "yaml_special_names": {
        "agent.py": (
            "from langchain_core.tools import tool\nfrom langchain_openai import ChatOpenAI\n"
            "@tool('look: up #1')\ndef f(q: str) -> str:\n    '''Look.'''\n    return q\n"
            "@tool(\"{weird}, [name] & *alias\")\ndef g(q: str) -> str:\n    '''G.'''\n    return q\n"
            "llm = ChatOpenAI(model='gpt-4o-2024-08-06').bind_tools([f, g])\n"
        )
    },
    "dynamic_tools": {
        "agent.py": (
            "import json\nfrom crewai.tools import tool\nfrom crewai import Agent\n"
            "def build():\n    out = []\n    for n, fn in json.load(open('t.json')).items():\n"
            "        out.append(tool(n)(fn))\n    return out\n"
            "a = Agent(role='r', goal='g', backstory='b', tools=build(), llm='gpt-4o')\n"
        )
    },
    "imports_missing_deps": {
        "agent.py": (
            "import framework_that_is_not_installed  # init must not import this\nopen('IMPORTED', 'w').write('x')\n"
        )
    },
}


def _make(tmp: Path, files: dict[str, str | bytes]) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        p = tmp / name
        if isinstance(body, bytes):
            p.write_bytes(body)
        else:
            p.write_text(body, encoding="utf-8")
    return tmp


def _init(path: Path, out: Path, *extra: str) -> tuple[int, str]:
    r = runner.invoke(app, ["init", str(path), "--out", str(out), *extra])
    return r.exit_code, r.output


def _assert_valid_scaffold(out: Path) -> None:
    loaded = load_spec(out)
    assert loaded.status == "incomplete", [i.message for i in loaded.issues]
    assert not loaded.invalid, [i.message for i in loaded.invalid]  # zero shape errors, ever
    assert [i for i in loaded.issues if i.kind != "invalid"] == []  # no prose / placeholders either
    tops = {str(p[0]) for p in loaded.todos}
    assert {"risk_level", "goal", "prohibited", "limits", "escalation", "owner"} <= tops  # user decisions only
    assert isinstance(loaded.raw.get("_derived"), dict)
    assert runner.invoke(app, ["validate", str(out)]).exit_code == 0  # init output passes validate


@pytest.mark.parametrize("agent", sorted(p.name for p in AGENTS.iterdir() if p.is_dir()))
def test_init_on_every_fixture_agent(tmp_path: Path, agent: str) -> None:
    code, _ = _init(AGENTS / agent, tmp_path / "spec.yml")
    assert code == 0
    _assert_valid_scaffold(tmp_path / "spec.yml")


@pytest.mark.parametrize("case", sorted(HOSTILE))
def test_init_on_hostile_code(tmp_path: Path, case: str) -> None:
    src = _make(tmp_path / "agent", HOSTILE[case]) if HOSTILE[case] else (tmp_path / "agent")
    src.mkdir(exist_ok=True)
    code, out = _init(src, tmp_path / "spec.yml")
    assert code == 0, out
    _assert_valid_scaffold(tmp_path / "spec.yml")
    derived = load_spec(tmp_path / "spec.yml").raw["_derived"]
    if case == "unknown_framework":
        assert derived["framework"] == "not recognised"
        assert "no recognised agent framework" in (tmp_path / "spec.yml").read_text(encoding="utf-8")
    if case in ("syntax_error", "null_bytes"):
        assert any("agent.py" in u for u in derived["unresolved"])  # what could not be read is listed
    if case == "yaml_special_names":
        assert set(derived["declared_tools"]) == {"look: up #1", "{weird}, [name] & *alias"}
    if case == "dynamic_tools":
        assert any("dynamic name / callable" in u or "runtime value" in u for u in derived["unresolved"])
    if case == "imports_missing_deps":
        assert not (src / "IMPORTED").exists()  # parsed, never imported or run


def test_init_falls_back_when_the_agent_cannot_run(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("import nothing_here\n", encoding="utf-8")
    # no mocks file: the mock run refuses; init must still write a valid spec from the code
    code, out = _init(
        tmp_path, tmp_path / "spec.yml", "--entry", str(tmp_path / "agent.py"), "--mocks", str(tmp_path / "mocks.yml")
    )
    flat = " ".join(out.split())
    assert code == 0 and "could not start" in flat and "mock-run needs a mocks file" in flat
    _assert_valid_scaffold(tmp_path / "spec.yml")
    assert "could not start" in (tmp_path / "spec.yml").read_text(encoding="utf-8")


def test_init_with_agent_id_and_derived_unresolved_is_yaml(tmp_path: Path) -> None:
    _init(AGENTS / "langgraph_po", tmp_path / "spec.yml", "--agent-id", "po: agent #1")
    raw = yaml.safe_load((tmp_path / "spec.yml").read_text(encoding="utf-8"))
    assert raw["agent_id"] == "po: agent #1"
    assert isinstance(raw["_derived"]["unresolved"], list)


# ------------------------------------------------------------------ params: every field, shape + example
def test_params_json_has_shape_and_a_valid_example_for_every_field() -> None:
    from agenticreq.models.spec import AgentSpec

    r = runner.invoke(app, ["params", "-f", "json"])
    rows = {d["name"]: d for d in json.loads(r.stdout)}
    for key in spec_fields():
        assert key in rows, key
        assert rows[key]["shape"] == shape(key)
        # the example is correct once each placeholder (<number>, <duration>) is replaced by a value of that kind
        AgentSpec.model_validate(_fill(yaml.safe_load(rows[key]["example_yaml"])))
        assert rows[key]["example"] == json.loads(json.dumps(example(key), default=str))


def _fill(v: object) -> object:
    """An example with its placeholders replaced by a value of that kind (a number the user would write)."""
    if v == "<number>":
        return 1
    if v == "<duration>":
        return "1h"
    if isinstance(v, dict):
        return {k: _fill(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_fill(x) for x in v]
    return v


def test_params_console_prints_every_field() -> None:
    out = runner.invoke(app, ["params"]).output
    assert out.count("example:") == len(spec_fields())
    assert "limit: {min: <number>, max: <number>}" in out


def test_params_one_field_and_a_typo() -> None:
    out = runner.invoke(app, ["params", "--field", "escalation"]).output
    assert "metric: refund_amount" in out and "shape:" in out
    r = runner.invoke(app, ["params", "--field", "escalaton"])
    assert r.exit_code == 2 and "escalation" in r.output
