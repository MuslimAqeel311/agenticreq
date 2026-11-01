"""The four result groups, sign-off expiry, the mock run (mocks / mock-run / check with no arguments), the removed
sandbox and discover, the renamed `confirmed:`, plain words in every output, and the terminal layout."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import Finding, PassedCheck, Report
from agenticreq.models.verdict import GROUP_OF, GROUPS, exit_code, overall, results, totals
from agenticreq.observe import build_observed
from agenticreq.report import render_console, render_json, render_sarif

ROOT = Path(__file__).parent.parent
FIX = Path(__file__).parent / "fixtures"
APP = FIX / "agents_misc" / "mock_run_app"
LANGGRAPH_APP = FIX / "agents_misc" / "finance_like"
runner = CliRunner()
#: "Unknown field" is the one allowed use of "unknown": the error for a wrong field name in spec.yml
BANNED = re.compile(
    r"attestation|attested|unknown(?! field)|not proven|heuristic|insufficient[ _]evidence|"
    r"sandbox(?!_policy)|discover|container|fakes?",
    re.IGNORECASE,
)
#: the one place an old word is allowed: the message that tells the user to rename `attestations:`
RENAME_MESSAGE = "`attestations:` was renamed to `confirmed:`; rename the key in spec.yml"


def _meta() -> dict[str, object]:
    return {"analysis": {"files": 1, "framework": "openai-agents", "tools": 1, "agents": 1}}


def _f(rule: str, conf: Confidence, severity: str = "warning", title: str = "t", **kw: object) -> Finding:
    return Finding(
        rule_id=rule,
        severity=severity,
        confidence=conf,
        title=title,
        detail="d",  # type: ignore[arg-type]
        evidence="e",
        **kw,
    )  # type: ignore[arg-type]


GAP = _f("AS-203", Confidence.proven(), "blocker", "Prohibited tool is reachable in code")
SIGN_OFF = _f("AS-501", Confidence.attestation(), "attestation", "'owner' is declared, but code cannot show it")
SIGNED = _f("AS-502", Confidence.attestation(), "attestation", "escalation signed by lead", reason="Signed by lead.")
CHECK = _f("AS-201", Confidence.unknown(), "info", "lookup never reached the model")
PASS = PassedCheck(rule_id="AS-210", subject="gpt-4o-2024-08-06", confidence=Confidence.proven())
CODE_PATTERN = PassedCheck(rule_id="AS-219", subject="tool error handling", confidence=Confidence.heuristic())


# =========================================================================== PART 1: the four groups
@pytest.mark.parametrize(
    "item,group,status",
    [
        (GAP, "gap", "GAP"),
        (SIGN_OFF, "sign_off", "SIGN-OFF"),
        (SIGNED, "sign_off", "SIGNED"),
        (CHECK, "mock_run", "CHECK"),
        (CODE_PATTERN, "mock_run", "CHECK"),
        (PASS, "passed", "PASS"),
    ],
    ids=["gap", "sign_off", "signed", "mock_run", "code_pattern", "passed"],
)
def test_each_result_lands_in_one_group(item: Finding | PassedCheck, group: str, status: str) -> None:
    rep = Report(
        agent_id="a",
        findings=[item] if isinstance(item, Finding) else [],
        passed=[item] if isinstance(item, PassedCheck) else [],
        metadata=_meta(),
    )
    (ln,) = results(rep)
    assert (ln.group, ln.status) == (group, status)
    d = json.loads(render_json(rep))
    assert (d["results"][0]["group"], d["results"][0]["status"]) == (group, status)
    if group == "mock_run":
        assert ln.reason.endswith("agenticreq mock-run")


def test_signed_is_never_counted_as_passed() -> None:
    rep = Report(agent_id="a", findings=[SIGNED], passed=[PASS], metadata=_meta())
    assert totals(results(rep)) == {
        "gaps": 0,
        "sign_off_left": 0,
        "signed": 1,
        "not_needed": 0,
        "mock_run": 0,
        "passed": 1,
    }


@pytest.mark.parametrize(
    "items,gaps,all_,never",
    [
        ([GAP], 1, 1, 0),
        ([SIGN_OFF], 0, 1, 0),
        ([CHECK], 0, 1, 0),
        ([SIGNED], 0, 0, 0),
        ([], 0, 0, 0),
    ],
    ids=["a_gap", "an_open_sign_off", "a_mock_run_check", "only_signed", "only_passed"],
)
def test_exit_code_per_fail_on_mode(items: list[Finding], gaps: int, all_: int, never: int) -> None:
    rep = Report(agent_id="a", findings=items, passed=[PASS], metadata=_meta())
    assert exit_code(rep, "gaps") == gaps and exit_code(rep, "all") == all_ and exit_code(rep, "never") == never
    assert overall(rep, fail_on="gaps") == ("FAIL" if gaps else "PASS")
    assert overall(rep, fail_on="all") == ("FAIL" if all_ else "PASS")


def test_invalid_spec_and_nothing_checked_keep_2_and_3(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("no code")
    for fail_on in ("gaps", "all"):
        assert runner.invoke(app, ["audit", str(tmp_path), "--fail-on", fail_on]).exit_code == 3
    spec = tmp_path / "spec.yml"
    spec.write_text("agent_id: x\nrisk_level: [bad]\n")
    assert runner.invoke(app, ["check", str(APP), "--spec", str(spec)]).exit_code == 2


def test_groups_print_in_order_and_empty_groups_are_hidden() -> None:
    rep = Report(agent_id="a", findings=[CHECK, GAP], passed=[PASS], metadata=_meta())
    out = render_console(rep)
    titles = [t for t in ("SPEC VS CODE GAPS", "HUMAN SIGN-OFF", "CHECK WITH A MOCK RUN", "PASSED") if t in out]
    assert titles == ["SPEC VS CODE GAPS", "CHECK WITH A MOCK RUN", "PASSED"]
    assert [out.index(t) for t in titles] == sorted(out.index(t) for t in titles)


def test_sarif_levels() -> None:
    rep = Report(agent_id="a", findings=[GAP, SIGN_OFF, SIGNED, CHECK], metadata=_meta())
    levels = {r["ruleId"]: r["level"] for r in json.loads(render_sarif(rep))["runs"][0]["results"]}
    assert levels == {"AS-203": "error", "AS-501": "note", "AS-502": "none", "AS-201": "note"}


SPEC = """agent_id: helpdesk
risk_level: medium
goal: {{statement: g, done_criteria: [d]}}
out_of_scope: []
prohibited: []
approval_required: []
limits: []
escalation: []
untrusted_inputs: []
owner: o@example.com
technical_owner: t@example.com
confirmed:
  - {{clause: owner, by: lead@example.com, date: {date}, evidence: "{evidence}", expires: {expires}}}
"""


def _owner(
    tmp: Path, *, date: str, expires: str, evidence: str = "T-1", today: dt.date = dt.date(2026, 10, 27)
) -> object:
    shutil.copytree(APP, tmp / "app", dirs_exist_ok=True)
    loaded = load_spec_text(SPEC.format(date=date, expires=expires, evidence=evidence))
    rep = run(mode="check", loaded=loaded, observed=build_observed(tmp / "app"), today=today)
    return next(ln for ln in results(rep) if ln.param == "owner")


def test_a_fresh_sign_off_is_signed(tmp_path: Path) -> None:
    ln = _owner(tmp_path, date="2026-10-01", expires="2099-01-01")
    assert (ln.status, ln.reason) == ("SIGNED", "Signed by lead@example.com on 2026-10-01.")  # type: ignore[attr-defined]


def test_an_expired_sign_off_is_open_again(tmp_path: Path) -> None:
    ln = _owner(tmp_path, date="2026-01-01", expires="2026-06-01")
    assert ln.status == "SIGN-OFF"  # type: ignore[attr-defined]
    assert "expired on 2026-06-01" in ln.reason  # type: ignore[attr-defined]


def test_a_sign_off_whose_evidence_file_changed_is_open_again(tmp_path: Path) -> None:
    ev = "helpdesk/tools.py:12"
    ln = _owner(tmp_path, date="2020-01-01", expires="2099-01-01", evidence=ev)  # the file is newer than the sign-off
    assert ln.status == "SIGN-OFF" and "file changed since sign-off" in ln.reason, ln  # type: ignore[attr-defined]
    tomorrow = (dt.date.today() + dt.timedelta(days=1)).isoformat()
    ln = _owner(tmp_path, date=tomorrow, expires="2099-01-01", evidence=ev)  # signed after the last change
    assert ln.status == "SIGNED"  # type: ignore[attr-defined]


def test_evidence_files_are_read_from_file_line_references() -> None:
    from agenticreq.signoff import evidence_files

    assert evidence_files("flag at graph.py:40 and src/app/flags.py:12; ticket T-9") == ["graph.py", "src/app/flags.py"]
    assert evidence_files("https://jira/T-1") == []


def test_a_sign_off_item_shows_what_the_code_found() -> None:
    found = PassedCheck(rule_id="AS-223", subject="env config", confidence=Confidence.heuristic(), param="kill_switch")
    so = _f("AS-501", Confidence.attestation(), "attestation", "'kill_switch' is declared, but code cannot show it")
    (ln,) = results(Report(agent_id="a", findings=[so], passed=[found], metadata=_meta()))
    assert ln.status == "SIGN-OFF" and ln.support.startswith("env config")
    out = render_console(Report(agent_id="a", findings=[so], passed=[found], metadata=_meta()))
    assert "Found: env config. Confirm it works under confirmed: in spec.yml." in out


# =========================================================================== PART 2: no sandbox, mocks required
def test_mock_run_without_a_mocks_file_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["mock-run", "--entry", "agent.py:main"])
    assert r.exit_code == 2
    assert "mock-run needs a mocks file: run `agenticreq mocks` to create mocks.yml" in " ".join(r.output.split())


def test_a_mock_run_still_hides_credentials_and_blocks_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agenticreq.capture import discover as mock_run

    monkeypatch.setenv("MY_SERVICE_TOKEN", "real-secret-value")
    (tmp_path / "agent.py").write_text(
        "import os, socket\n\ndef main():\n"
        "    assert 'MY_SERVICE_TOKEN' not in os.environ, 'a credential reached the agent'\n"
        "    assert os.environ['OPENAI_API_KEY'].startswith('agenticreq-fake-')\n"
        "    try:\n        socket.create_connection(('example.com', 80), timeout=3)\n"
        "    except OSError:\n        pass\n",
        encoding="utf-8",
    )
    (tmp_path / "mocks.yml").write_text("tools: {}\n", encoding="utf-8")
    res = mock_run(f"{tmp_path / 'agent.py'}:main", mock_tools=str(tmp_path / "mocks.yml"))
    assert not res.run_errors, res.run_errors  # neither assert fired
    assert "MY_SERVICE_TOKEN" in res.scrubbed_env
    assert any(b.startswith(("dns:example.com", "connect:")) for b in res.egress_blocked), res.egress_blocked


def test_sandbox_options_are_gone() -> None:
    for cmd in ("mock-run", "init"):
        r = runner.invoke(app, [cmd, "--sandbox", "sandbox.yml"])
        assert r.exit_code == 2 and "No such option" in r.output, cmd
    import agenticreq.capture as capture

    assert not hasattr(capture, "SandboxRefused")


# =========================================================================== CHANGE: discover is mock-run
def test_discover_is_gone_and_mock_run_is_listed() -> None:
    r = runner.invoke(app, ["discover", "agent.py"])
    assert r.exit_code == 2 and "No such command" in r.output
    top = runner.invoke(app, ["--help"], terminal_width=200).output
    assert "mock-run" in top and "mocks" in top and "discover" not in top


# =========================================================================== PART 3: mocks / mock-run / check
def _mocks_yaml(path: Path, *extra: str) -> dict:  # type: ignore[type-arg]
    r = runner.invoke(app, ["mocks", str(path), "--out", str(path / "mocks.yml"), *extra])
    assert r.exit_code == 0, r.output
    return yaml.safe_load((path / "mocks.yml").read_text(encoding="utf-8"))


def test_mocks_reads_the_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(APP, tmp_path / "app")
    (tmp_path / "app" / "spec.yml").unlink()
    monkeypatch.chdir(tmp_path / "app")
    data = _mocks_yaml(tmp_path / "app")
    assert data["entry"] == "helpdesk.agents:run_agent"
    assert data["args"] == {"name": "support", "task": "What can you help me with?"}  # a key of TOOLSETS
    assert data["python"] is None  # detected at run time
    assert data["order"] == ["lookup_order", "lookup_customer", "send_refund"]
    assert set(data["tools"]) == {"lookup_order", "lookup_customer", "send_refund"}  # every tool found
    assert data["tools"]["lookup_order"] == {"args": {"order_id": "1"}}  # read-only: sample args
    assert data["tools"]["send_refund"] == {
        "patch": "helpdesk.tools:send_refund",
        "returns": "mock result for send_refund",
    }  # a write: never runs
    text = (tmp_path / "app" / "mocks.yml").read_text(encoding="utf-8")
    assert text.startswith("# mocks.yml") and "# entry:" in text and "# tools:" in text


def test_mocks_gives_prohibited_tools_call_false_and_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shutil.copytree(APP, tmp_path / "app")
    monkeypatch.chdir(tmp_path / "app")  # ./spec.yml prohibits send_refund
    data = _mocks_yaml(tmp_path / "app")
    assert data["tools"]["send_refund"] == {"call": False}
    r = runner.invoke(app, ["mocks", str(tmp_path / "app"), "--out", str(tmp_path / "app" / "mocks.yml")])
    assert r.exit_code == 2 and "refusing to overwrite" in r.output
    assert _mocks_yaml(tmp_path / "app", "--force")["tools"]["send_refund"] == {"call": False}


def test_mocks_writes_todo_when_no_entry_is_found(tmp_path: Path) -> None:
    (tmp_path / "tools.py").write_text(
        "from langchain_core.tools import tool\n@tool\ndef ping(x: str) -> str:\n    '''Ping.'''\n    return x\n",
        encoding="utf-8",
    )
    data = _mocks_yaml(tmp_path)
    assert data["entry"] == "TODO" and data["args"] == "TODO"
    assert "module:function" in (tmp_path / "mocks.yml").read_text(encoding="utf-8")  # the comment says how


def test_mocks_on_a_langgraph_multi_agent_app_picks_an_offline_agent(tmp_path: Path) -> None:
    shutil.copytree(LANGGRAPH_APP, tmp_path / "app")
    data = _mocks_yaml(tmp_path / "app")
    assert data["entry"] == "finance_agents.agents:run_agent"
    assert data["args"]["name"] == "financial_planner"  # its tools use no outside service (the others call yfinance)
    assert set(data["tools"]) == {
        "get_stock_quote",
        "compare_tickers",
        "loan_amortization",
        "estimate_tax",
        "create_order",
    }
    assert "patch" in data["tools"]["create_order"]  # a write tool never runs
    assert "yfinance" in (tmp_path / "app" / "mocks.yml").read_text(encoding="utf-8")


def test_mocks_then_mock_run_then_check_with_no_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(APP, tmp_path / "app")
    monkeypatch.chdir(tmp_path / "app")
    assert runner.invoke(app, ["mocks"]).exit_code == 0
    r = runner.invoke(app, ["mock-run"])
    out = " ".join(r.output.split())
    assert r.exit_code == 0, out
    assert "captured 3/3 LLM calls" in out and "tools offered to the model: lookup_order, lookup_customer" in out
    assert "not asked for (call: false in mocks.yml): send_refund" in out  # prohibited in spec.yml
    # call: false only means "not called": a tool no run reached is still listed, with both facts
    assert "never offered to the model: send_refund" in out and "call: false" in out
    assert "saved: " in out and (tmp_path / "app" / ".agenticreq" / "captures.json").is_file()
    c = runner.invoke(app, ["check", "-f", "json"])
    data = json.loads(c.stdout)
    assert data["metadata"]["mock_run"]["captured"] == 3 and data["metadata"]["mock_run"]["stale"] is False
    params = {r["param"] for r in data["results"]}
    assert "runtime_evidence" not in params  # "the agent has not run yet" is gone: the mock run counts
    refund = next(r for r in data["results"] if r["param"] == "send_refund")  # never offered: call: false
    assert refund["status"] == "CHECK" and refund["reason"].endswith("agenticreq mock-run")


def test_mock_run_exits_1_when_the_run_crashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "agent.py").write_text(
        "def run(question: str) -> str:\n    raise RuntimeError('boom')\n", encoding="utf-8"
    )
    (tmp_path / "mocks.yml").write_text("entry: agent.py:run\nargs: {question: hi}\ntools: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["mock-run"])
    assert r.exit_code == 1 and "stopped at agent.py:2 with RuntimeError: boom" in " ".join(r.output.split())


def test_flags_override_mocks_yml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "agent.py").write_text(
        "def run(question: str) -> str:\n    raise SystemExit(0 if question == 'x' else 4)\n", encoding="utf-8"
    )
    (tmp_path / "mocks.yml").write_text("entry: TODO\nargs: TODO\ntools: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["mock-run"])
    assert r.exit_code == 2 and "has no entry" in r.output
    r = runner.invoke(app, ["mock-run", "--entry", "agent.py:run", "--args", '{"question": "x"}'])
    assert "crashed" not in r.output  # the flags reached the entry (SystemExit(4) would be a crash)


@pytest.mark.parametrize(
    "osname,parts", [("nt", (".venv", "Scripts", "python.exe")), ("posix", (".venv", "bin", "python"))]
)
def test_python_is_found_in_the_local_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, osname: str, parts: tuple[str, ...]
) -> None:
    from agenticreq import mockgen

    exe = tmp_path.joinpath(*parts)
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(mockgen.os, "name", osname)
    assert mockgen.detect_python(tmp_path) == (str(exe), "./.venv")
    assert mockgen.detect_python(tmp_path, "py3") == ("py3", "python: in mocks.yml")  # label renamed (pyenv.py)
    shutil.rmtree(tmp_path / ".venv")
    assert mockgen.detect_python(tmp_path) == (sys.executable, "agenticreq's own interpreter")  # label renamed


def test_a_patch_replaces_the_function_inside_a_tool_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agenticreq.capture.responder import MockConfig
    from agenticreq.capture.runner import apply_patches

    shutil.copytree(APP, tmp_path / "app")
    monkeypatch.syspath_prepend(str(tmp_path / "app"))
    (tmp_path / "m.yml").write_text(
        'tools:\n  send_refund: {patch: "helpdesk.tools:send_refund", returns: ok}\n', encoding="utf-8"
    )
    apply_patches(MockConfig.load(tmp_path / "m.yml"))
    import helpdesk.tools as tools

    assert tools.BILLING_TOOLS[1].func(order_id="A1", amount=5.0) == "ok"  # the list built at import got the stub
    for mod in [m for m in sys.modules if m == "helpdesk" or m.startswith("helpdesk.")]:
        del sys.modules[mod]


# =========================================================================== PART 4: plain words
def test_old_attestations_key_stops_with_a_clear_message(tmp_path: Path) -> None:
    spec = tmp_path / "spec.yml"
    spec.write_text(
        (APP / "spec.yml").read_text(encoding="utf-8") + "attestations:\n  - {clause: owner, by: lead@example.com}\n",
        encoding="utf-8",
    )
    r = runner.invoke(app, ["check", str(APP), "--spec", str(spec)], terminal_width=200)
    assert r.exit_code == 2 and RENAME_MESSAGE in " ".join(r.output.split())


def _fixture_dirs() -> list[Path]:
    out = []
    for group in ("agents", "agents_crewai", "agents_misc", "passfail", "runtime_proof"):
        out += [d for d in sorted((FIX / group).iterdir()) if d.is_dir() and d.name != "__pycache__"]
    return [*out, FIX / "spike"]


@pytest.mark.parametrize("d", _fixture_dirs(), ids=lambda d: f"{d.parent.name}/{d.name}")
def test_no_banned_word_in_any_output(d: Path) -> None:
    spec = d / "spec.yml" if (d / "spec.yml").is_file() else d.parent / "spec.yml"
    spec = spec if spec.is_file() else FIX / "specs" / "good.yml"
    for cmd in ("audit", "check"):
        for fmt in ("console", "json", "sarif"):
            for verbose in [[], ["--verbose"]] if fmt == "console" else [[]]:
                args = [cmd, str(d), "-f", fmt, *verbose] + (["--spec", str(spec)] if cmd == "check" else [])
                out = runner.invoke(app, args, terminal_width=200).output
                hits = [ln for ln in out.splitlines() if BANNED.search(ln)]
                assert not hits, (cmd, fmt, verbose, hits[:3])


def test_no_banned_word_in_help_readme_or_docs() -> None:
    top = runner.invoke(app, ["--help"], terminal_width=200).output
    commands = re.findall(r"^\W+([a-z][a-z-]+)\s", top, re.M)
    assert "mock-run" in commands
    for c in [None, *dict.fromkeys(commands)]:
        out = runner.invoke(app, [*([c] if c else []), "--help"], terminal_width=200).output
        assert not BANNED.search(out), (c, BANNED.findall(out))
    for doc in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        text = doc.read_text(encoding="utf-8")
        assert not BANNED.search(text), (doc.name, BANNED.findall(text)[:5])


# =========================================================================== PART 5: terminal layout
GOLDEN = Path(__file__).parent / "golden"
_TIME = re.compile(r"mock run: \d{4}-\d\d-\d\d \d\d:\d\d")


def _snapshot(d: Path, monkeypatch: pytest.MonkeyPatch, *args: str) -> str:
    monkeypatch.chdir(d)
    monkeypatch.setenv("COLUMNS", "100")
    out = runner.invoke(app, list(args), terminal_width=100).output
    return _TIME.sub("mock run: <time>", out)


@pytest.mark.parametrize(
    "name,where,args",
    [
        ("single_agent", FIX / "passfail" / "clean", ("check", ".", "--spec", "spec.yml", "--quiet")),
        ("multi_agent", FIX / "passfail" / "two_agents", ("audit", ".", "--quiet")),
        (
            "all_passing",
            FIX / "passfail" / "captured",
            ("check", ".", "--spec", "spec.yml", "--captures", "captures.json", "--fail-on", "all", "--quiet"),
        ),
    ],
)
def test_golden_terminal_output(name: str, where: Path, args: tuple[str, ...], monkeypatch: pytest.MonkeyPatch) -> None:
    got = _snapshot(where, monkeypatch, *args)
    path = GOLDEN / f"{name}.txt"
    if os.environ.get("UPDATE_GOLDEN") or not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_text(got, encoding="utf-8")
    assert got == path.read_text(encoding="utf-8")


def test_golden_layout_rules() -> None:
    for name in ("single_agent", "multi_agent", "all_passing"):
        lines = [ln for ln in (GOLDEN / f"{name}.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert lines[0].startswith("agenticreq ") and lines[1].startswith("Result: "), name
        assert all(len(ln) <= 100 for ln in lines), name
    assert "Result: PASS" in (GOLDEN / "all_passing.txt").read_text(encoding="utf-8")
    assert "[reporter]" in (GOLDEN / "multi_agent.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "env",
    [{"COLUMNS": "60"}, {"COLUMNS": "120"}, {"NO_COLOR": "1"}, {"PYTHONIOENCODING": "cp1252"}],
    ids=lambda e: "-".join(f"{k}={v}" for k, v in e.items()),
)
def test_any_width_colour_or_encoding(env: dict[str, str]) -> None:
    full = {**os.environ, "COLUMNS": "100", "PYTHONIOENCODING": "utf-8", **env}
    full.pop("NO_COLOR", None) if "NO_COLOR" not in env else None
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "agenticreq.cli",
            "check",
            str(FIX / "passfail" / "clean"),
            "--spec",
            str(FIX / "passfail" / "clean" / "spec.yml"),
        ],
        capture_output=True,
        env=full,
    )
    assert p.returncode in (0, 1), p.stderr.decode(errors="replace")
    enc = full["PYTHONIOENCODING"]
    text = p.stdout.decode(enc)
    width = int(full["COLUMNS"])
    assert all(len(ln) <= width for ln in text.splitlines()), [ln for ln in text.splitlines() if len(ln) > width][:2]
    assert "\x1b[" not in text  # not a terminal: no colour codes
    if enc == "cp1252":
        assert "✓" not in text and "✗" not in text and "…" not in text and "PASSED" in text
        assert re.search(r"^  [+x?] ", text, re.M)
    else:
        assert "✓" in text and "·" in text.splitlines()[0]


# =========================================================================== PART 6: check reads the mock run
def _ran_app(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copytree(APP, tmp / "app")
    monkeypatch.chdir(tmp / "app")
    assert runner.invoke(app, ["mocks"]).exit_code == 0
    assert runner.invoke(app, ["mock-run"]).exit_code == 0
    return tmp / "app"


def _rows_of(out: str) -> list[str]:
    """Report rows with their wrapped continuation lines joined (a reason is never cut, it wraps)."""
    rows: list[str] = []
    for ln in out.splitlines():
        if ln.startswith("     ") and rows and ln.strip():
            rows[-1] += " " + ln.strip()
        else:
            rows.append(ln)
    return rows


def test_check_without_a_mock_run_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shutil.copytree(APP, tmp_path / "app")
    monkeypatch.chdir(tmp_path / "app")
    out = runner.invoke(app, ["check", "--all"], terminal_width=200).output  # --all: the spec still has gaps
    assert out.splitlines()[0].endswith("no mock run yet")
    checks = [" ".join(row.split()) for row in _rows_of(out) if row.startswith("  ? ")]
    assert checks and all(ln.endswith("agenticreq mock-run") for ln in checks)


def test_check_reads_the_latest_mock_run_and_hides_its_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _ran_app(tmp_path, monkeypatch)
    out = runner.invoke(app, ["check"], terminal_width=200).output
    assert re.search(r"mock run: \d{4}-\d\d-\d\d \d\d:\d\d \(3/3 calls\)", out.splitlines()[0])
    assert "captures.json" not in out and "Code changed after" not in out
    verbose = runner.invoke(app, ["check", "--verbose"], terminal_width=200).output
    assert str(d / ".agenticreq" / "captures.json") in verbose or ".agenticreq" in verbose


def test_a_stale_mock_run_is_flagged_and_left_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _ran_app(tmp_path, monkeypatch)
    later = (d / ".agenticreq" / "captures.json").stat().st_mtime + 120
    os.utime(d / "helpdesk" / "tools.py", (later, later))
    out = runner.invoke(app, ["check"], terminal_width=200).output
    assert "its results are left out of this report" in " ".join(out.split())
    data = json.loads(runner.invoke(app, ["check", "-f", "json"]).stdout)
    assert data["metadata"]["mock_run"]["stale"] is True
    assert "runtime_evidence" in {r["param"] for r in data["results"]}  # an out-of-date run is not evidence


def test_captures_flag_overrides_the_latest_mock_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = _ran_app(tmp_path, monkeypatch)
    other = tmp_path / "other.json"
    shutil.copy(d / ".agenticreq" / "captures.json", other)
    (d / ".agenticreq" / "captures.json").unlink()
    data = json.loads(runner.invoke(app, ["check", "--captures", str(other), "-f", "json"]).stdout)
    assert data["metadata"]["mock_run"]["file"] == str(other)
    assert GROUP_OF["CHECK"] == "mock_run" and GROUPS[0] == "gap"
