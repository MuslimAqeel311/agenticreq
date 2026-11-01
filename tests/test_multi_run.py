"""Several mock runs in one mocks.yml (`runs:`), `--run`, `--keep`, the merged captures, and call: false."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from agenticreq.capture.discover import DiscoverError, plan_runs
from agenticreq.capture.responder import MockConfig
from agenticreq.cli import app

FIX = Path(__file__).parent / "fixtures" / "agents_misc"
runner = CliRunner()
BANNED = re.compile(r"\bfakes?\b|sandbox|attestation|unknown", re.IGNORECASE)


def _text(r) -> str:  # type: ignore[no-untyped-def]
    return " ".join(r.output.split())


def _app(tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    dest = tmp_path / "app"
    shutil.copytree(FIX / name, dest)
    monkeypatch.chdir(dest)
    return dest


def _saved(root: Path) -> dict:  # type: ignore[type-arg]
    return json.loads((root / ".agenticreq" / "captures.json").read_text(encoding="utf-8"))


def _calls_per_run(data: dict) -> dict[str, int]:  # type: ignore[type-arg]
    out: dict[str, int] = {}
    for c in data["calls"]:
        out[c["run_id"]] = out.get(c["run_id"], 0) + 1
    return out


# ============================================================================ mocks.yml: the config
def _cfg(tmp_path: Path, body: str) -> MockConfig:
    p = tmp_path / "mocks.yml"
    p.write_text(body, encoding="utf-8")
    return MockConfig.load(p)


def test_old_single_task_file_has_no_runs(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "entry: agent.py:run\nargs: {task: x}\ntools: {t: {args: {a: 1}}}\n")
    assert cfg.runs == [] and cfg.for_run(None) is cfg and cfg.for_run("anything") is cfg
    assert plan_runs(cfg) == [("run-0", None)]
    assert [rid for rid, _ in plan_runs(cfg, repeat=3)] == ["run-0", "run-1", "run-2"]  # --runs N keeps its ids


def test_runs_are_read_in_file_order_with_their_own_inputs(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path, "entry: a.py:f\nruns:\n  - {name: one, args: {q: 1}}\n  - {name: two, entry: b.py:g}\n  - {}\n"
    )
    assert cfg.run_names() == ["one", "two", "run-2"]  # a run with no name is run-<position>
    assert cfg.runs[0].has_args and cfg.runs[0].args == {"q": 1}
    assert cfg.runs[1].entry == "b.py:g" and not cfg.runs[1].has_args
    assert [rid for rid, _ in plan_runs(cfg)] == ["one", "two", "run-2"]
    assert [rid for rid, _ in plan_runs(cfg, repeat=2, only=["two"])] == ["two", "two#2"]


@pytest.mark.parametrize(
    "body,needle",
    [
        ("runs: x\n", "runs: must be a list"),
        ("runs: [1]\n", "runs[0] must be a mapping"),
        ("runs:\n  - {name: a}\n  - {name: a}\n", "two runs are named 'a'"),
        ("runs:\n  - {name: a, arg: 1}\n", "run 'a' has unknown field arg"),
        ("runs:\n  - {name: a, tools: [x]}\n", "tools: must be a mapping"),
    ],
)
def test_a_wrong_runs_entry_is_named(tmp_path: Path, body: str, needle: str) -> None:
    with pytest.raises(ValueError) as e:
        _cfg(tmp_path, body)
    assert needle in str(e.value)


def test_a_run_changes_order_and_tool_args_without_touching_the_file_level(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        "order: [a, b]\ntools:\n  a: {args: {x: 1}}\n  b: {call: false}\nstructured: {k: 1}\n"
        "runs:\n  - name: r\n    order: [b]\n    structured: {j: 2}\n"
        "    tools:\n      a: {args: {y: 2}}\n      b: {call: true}\n      c: {args: {z: 3}}\n",
    )
    r = cfg.for_run("r")
    assert r.order == ["b"] and r.structured == {"k": 1, "j": 2}
    assert r.tools["a"].args == {"x": 1, "y": 2} and r.tools["b"].call is True and r.tools["c"].args == {"z": 3}
    assert cfg.order == ["a", "b"] and cfg.tools["a"].args == {"x": 1} and cfg.tools["b"].call is False
    assert "c" not in cfg.tools  # the file level is unchanged


def test_picking_a_run_that_does_not_exist_says_which_exist(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "runs:\n  - {name: a}\n  - {name: b}\n")
    with pytest.raises(DiscoverError, match=r"no mock run named c .*its runs are: a, b"):
        plan_runs(cfg, only=["c"])
    with pytest.raises(DiscoverError, match="has no runs"):
        plan_runs(_cfg(tmp_path, "tools: {}\n"), only=["a"])


# ============================================================================ the CLI, no framework needed
def test_three_runs_are_all_captured_and_summarised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _app(tmp_path, "multi_run_app", monkeypatch)
    r = runner.invoke(app, ["mock-run"])
    out = _text(r)
    assert r.exit_code == 0, out
    for name, tool in (("billing", "get_invoice"), ("support", "open_ticket"), ("sales", "quote_price")):
        assert f"mock run {name}: 2 LLM calls: {tool}" in out  # one line per run
    assert "captured 6/6 LLM calls in 3 run(s)" in out
    assert "tools offered to the model: get_invoice, open_ticket, quote_price" in out  # the combined summary
    assert "never offered" not in out and not BANNED.search(out)
    data = _saved(root)
    assert data["run_ids"] == ["billing", "support", "sales"] and _calls_per_run(data) == {
        "billing": 2,
        "support": 2,
        "sales": 2,
    }


def test_one_run_is_the_old_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _app(tmp_path, "multi_run_app", monkeypatch)
    (root / "mocks.yml").write_text(
        "entry: agent.py:run\nargs: {task: support}\ntools:\n  open_ticket: {args: {ref: t}}\n", encoding="utf-8"
    )  # the old format: no runs:
    out = _text(runner.invoke(app, ["mock-run"]))
    assert "captured 2/2 LLM calls in 1 run(s)" in out and "tools offered to the model: open_ticket" in out
    assert "mock run " not in out  # no per-run lines for a file without runs:
    assert _saved(root)["run_ids"] == ["run-0"]


def test_flags_still_override_each_runs_own_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _app(tmp_path, "multi_run_app", monkeypatch)
    out = _text(runner.invoke(app, ["mock-run", "--args", '{"task": "sales"}']))
    assert "tools offered to the model: quote_price" in out and "captured 6/6 LLM calls in 3 run(s)" in out
    assert _calls_per_run(_saved(root)) == {"billing": 2, "support": 2, "sales": 2}


def test_run_flag_picks_runs_and_replaces_the_file_without_keep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _app(tmp_path, "multi_run_app", monkeypatch)
    assert runner.invoke(app, ["mock-run"]).exit_code == 0
    out = _text(runner.invoke(app, ["mock-run", "--run", "support"]))
    assert "captured 2/2 LLM calls in 1 run(s)" in out and "tools offered to the model: open_ticket" in out
    assert "never offered to the model: get_invoice, quote_price:" in out
    assert _saved(root)["run_ids"] == ["support"]  # overwrite: the other two runs are gone
    bad = runner.invoke(app, ["mock-run", "--run", "nope"])
    assert bad.exit_code == 2 and "no mock run named nope" in _text(bad) and "billing, support, sales" in _text(bad)


def test_keep_merges_and_a_run_run_again_replaces_only_itself(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _app(tmp_path, "multi_run_app", monkeypatch)
    assert runner.invoke(app, ["mock-run", "--run", "billing"]).exit_code == 0
    out = _text(runner.invoke(app, ["mock-run", "--run", "sales", "--keep"]))  # merge
    assert (
        "mock run billing (kept): 2 LLM calls: get_invoice" in out and "mock run sales: 2 LLM calls: quote_price" in out
    )
    assert "captured 4/4 LLM calls in 2 run(s)" in out and "tools offered to the model: get_invoice, quote_price" in out
    out = _text(runner.invoke(app, ["mock-run", "--run", "billing", "--run", "support", "--keep"]))
    data = _saved(root)
    assert _calls_per_run(data) == {"billing": 2, "sales": 2, "support": 2}  # billing replaced, not doubled
    assert sorted(data["run_ids"]) == ["billing", "sales", "support"] and "(kept)" in out
    assert [c["index"] for c in data["calls"]] == list(range(6))  # one numbering over the merged file
    assert "never offered" not in out and data["coverage"] == "6/6"
    out = _text(runner.invoke(app, ["mock-run", "--run", "sales", "--keep"]))  # overwrite one run again
    assert _calls_per_run(_saved(root)) == {"billing": 2, "support": 2, "sales": 2} and "captured 6/6" in out


def test_keep_with_no_earlier_file_or_an_unreadable_one_just_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _app(tmp_path, "multi_run_app", monkeypatch)
    assert runner.invoke(app, ["mock-run", "--run", "sales", "--keep"]).exit_code == 0
    assert _saved(root)["run_ids"] == ["sales"]
    (root / ".agenticreq" / "captures.json").write_text("{not json", encoding="utf-8")
    assert runner.invoke(app, ["mock-run", "--run", "billing", "--keep"]).exit_code == 0
    assert _saved(root)["run_ids"] == ["billing"]


def test_keep_reads_a_file_saved_before_runs_were_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _app(tmp_path, "multi_run_app", monkeypatch)
    assert runner.invoke(app, ["mock-run", "--run", "support"]).exit_code == 0
    p = root / ".agenticreq" / "captures.json"
    old = json.loads(p.read_text(encoding="utf-8"))
    for k in ("run_ids", "by_run"):
        old.pop(k)  # what an older agenticreq wrote
    p.write_text(json.dumps(old), encoding="utf-8")
    out = _text(runner.invoke(app, ["mock-run", "--run", "sales", "--keep"]))
    assert "captured 4/4 LLM calls in 2 run(s)" in out and _saved(root)["run_ids"] == ["support", "sales"]


def test_call_false_does_not_hide_a_tool_no_run_reached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _app(tmp_path, "multi_run_app", monkeypatch)
    cfg = yaml.safe_load((root / "mocks.yml").read_text(encoding="utf-8"))
    cfg["tools"]["quote_price"] = {"call": False}
    (root / "mocks.yml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = _text(runner.invoke(app, ["mock-run", "--run", "billing"]))
    assert "not asked for (call: false in mocks.yml): quote_price" in out
    line = next(
        x
        for x in re.split(r"(?=never offered to the model:)", out)
        if x.startswith("never offered to the model: quote_price")
    )
    assert "call: false" in line and "never asked to call it" in line  # both facts, in one line
    # once a run reaches it, call: false only means "not called": it is offered and no longer unreached
    out = _text(runner.invoke(app, ["mock-run"]))
    assert "tools offered to the model: get_invoice, open_ticket" in out and "never offered" not in out


def test_a_run_that_fails_is_reported_and_exits_1_while_the_others_are_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _app(tmp_path, "multi_run_app", monkeypatch)
    cfg = yaml.safe_load((root / "mocks.yml").read_text(encoding="utf-8"))
    cfg["runs"][1]["args"] = {"task": "missing"}  # KeyError inside the agent
    (root / "mocks.yml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    r = runner.invoke(app, ["mock-run"])
    out = _text(r)
    assert r.exit_code == 1 and "KeyError" in out and "mock run billing: 2 LLM calls" in out
    assert "mock run support: 0 LLM calls: no tools offered" in out
    assert _calls_per_run(_saved(root)) == {"billing": 2, "sales": 2}


# ============================================================================ the agent fixtures (OpenAI Agents SDK)
def _group(root: Path, group: str) -> list[str]:
    r = runner.invoke(app, ["check", "-f", "json"])
    return [x["param"] for x in json.loads(r.stdout)["results"] if x.get("group") == group]


def test_triage_with_three_specialists_is_fully_covered_by_three_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("agents")
    root = _app(tmp_path, "bank_triage", monkeypatch)
    out = _text(runner.invoke(app, ["mock-run"]))
    assert "mock run accounts" in out and "mock run fraud" in out and "mock run loans" in out
    assert "tools offered to the model:" in out and "never offered" not in out
    for tool in ("get_balances", "flag_transaction", "loan_estimate"):
        assert tool in out
    assert _group(root, "mock_run") == []  # 0 items under "Check with a mock run"
    # one run reaches one specialist only: the other two stay under "Check with a mock run", naming the agent
    out = _text(runner.invoke(app, ["mock-run", "--run", "fraud"]))
    assert "get_balances: the run never reached Accounts" in out and "loan_estimate: the run never reached Loans" in out
    assert "Add a run under runs: in mocks.yml" in out
    assert sorted(_group(root, "mock_run")) == ["get_balances", "loan_estimate"]
    # --keep adds the missing specialists without running fraud again
    runner.invoke(app, ["mock-run", "--run", "accounts", "--run", "loans", "--keep"])
    assert _group(root, "mock_run") == [] and _saved(root)["run_ids"] == ["fraud", "accounts", "loans"]


def test_an_agent_nothing_hands_off_to_stays_under_check_with_a_mock_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("agents")
    root = _app(tmp_path, "bank_unreachable", monkeypatch)
    out = _text(runner.invoke(app, ["mock-run"]))
    assert "never offered to the model: audit_log: the run never reached Audit" in out  # names the agent, not a tool
    assert "Add a run under runs: in mocks.yml" in out
    assert _group(root, "mock_run") == ["audit_log"]  # still shown after every run: it needs a user-written task
