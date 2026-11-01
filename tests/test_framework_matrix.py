"""Cross-framework verification: one small multi-agent project per framework (tests/fixtures/matrix, made by
make.py), each run through the whole path:

    audit -> init -> fill the spec -> check -> mocks -> mock-run -> check -> confirmed: entries -> check

The frameworks are not installed here, so the mock run is the captures such a run saves: one model call per agent,
offering that agent's tools under the names sent to the model (web_search for "Web Search"). Run this file directly
for the table: python tests/test_framework_matrix.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parent))
from conftest import chat_call  # noqa: E402

from agenticreq.ast.classify import canonical  # noqa: E402
from agenticreq.cli import app  # noqa: E402
from agenticreq.graph.origin import looks_generated  # noqa: E402
from agenticreq.observe import build_observed  # noqa: E402
from agenticreq.params import PARAMS  # noqa: E402
from agenticreq.progress import ALL_DONE  # noqa: E402
from agenticreq.scaffold import code_agents  # noqa: E402
from agenticreq.subfields import rule_of, sign_off_only  # noqa: E402

MATRIX = Path(__file__).parent / "fixtures" / "matrix"
FRAMEWORKS = [
    "crewai",
    "langgraph",
    "openai_agents",
    "pydantic_ai",
    "smolagents",
    "autogen",
    "llamaindex",
    "google_adk",
    "agno",
    "strands",
]
ASSERTIONS = [
    "1 init writes no names",
    "2 no name in two opposite lists",
    "3 no gap asks to register an id",
    "4 no needless mock-run line",
    "5 every sub-field checked or listed",
    "6 not_needed on every type",
    "7 nothing cut, coverage complete",
    "8 all 5 steps done",
]
runner = CliRunner()
RUN = "agenticreq mock-run"


def _invoke(args: list[str], width: int = 120) -> Any:
    return runner.invoke(app, args, terminal_width=width, env={"COLUMNS": str(width)})


def _json(args: list[str]) -> dict[str, Any]:
    return json.loads(_invoke([*args, "-f", "json", "--quiet"]).stdout)  # type: ignore[no-any-return]


def _rows(out: str) -> list[str]:
    rows: list[str] = []
    for ln in out.splitlines():
        if ln.startswith("     ") and rows and ln.strip():
            rows[-1] += " " + ln.strip()
        else:
            rows.append(ln)
    return rows


def _fill(spec: Path, d: Path) -> None:
    """What a person writes: their own requirements, with not_needed on a field of every type."""
    obs = build_observed(d)
    data = yaml.safe_load(spec.read_text(encoding="utf-8"))
    tools = sorted(obs.declared_tools)
    agents = code_agents(obs)
    edges = [e.split(" -> ") for e in data["_derived"].get("delegation_in_code") or []]
    reader = next(t for t in tools if "search" in canonical(t))
    data.update(
        {
            "agent_id": "research-team",
            "risk_level": "medium",
            "topology": "supervisor",
            "goal": {"statement": "write a sourced report", "done_criteria": ["the report is saved"]},
            "out_of_scope": [],
            "role_boundary": {"allowed_tools": tools},
            "prohibited": [],
            "limits": [],
            "untrusted_inputs": [{"source": reader}],
            "owner": "lead@example.com",
            "technical_owner": "dev@example.com",
            "agent_registry": [{"id": a} for a in agents],
            "declared_delegation": [{"from": a, "to": b} for a, b in edges],
            # not_needed on a list, a mapping, a string, a boolean, an enum and a nested number
            "approval_required": "not_needed",
            "escalation": "not_needed",
            "data_rules": "not_needed",
            "conflict_resolution": "not_needed",
            "goal_drift_guard": "not_needed",
            "tool_schema_strictness": "not_needed",
            "bounds": {"max_cost_usd": "not_needed"},
        }
    )
    spec.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _mock_run(d: Path) -> None:
    """The captures a mock run saves: one call per agent, its tools under their wire names."""
    obs = build_observed(d)
    calls = []
    for i, a in enumerate(x for x in obs.declared_agents if x.kind != "group_chat"):
        tools = [
            {"name": canonical(t), "description": "d", "parameters": {"type": "object", "properties": {}}}
            for t in a.tools
        ]
        calls.append(chat_call(i, tools, system=a.system_prompt or f"You are {a.name}.", synthetic=True))
    out = d / ".agenticreq" / "captures.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"calls": [c.model_dump(mode="json") for c in calls]}), encoding="utf-8")
    old = out.stat().st_mtime - 3600
    for p in d.rglob("*.py"):
        os.utime(p, (old, old))  # the mock run is newer than the code


def _confirm(spec: Path, report: dict[str, Any]) -> None:
    """The owner answers every open item: yes for a sign-off, a reason for not_needed, a waiver for a gap."""
    data = yaml.safe_load(spec.read_text(encoding="utf-8"))
    confirmed, waivers = [], []
    for r in report["results"]:
        if r["status"] == "SIGN-OFF":
            confirmed.append(
                {
                    "clause": r["param"],
                    "answer": "yes",
                    "by": "lead@example.com",
                    "date": "2026-10-01",
                    "expires": "2099-01-01",
                    "evidence": "reviewed by the team",
                }
            )
        elif r["status"] == "NOT NEEDED" and r.get("open"):
            confirmed.append(
                {
                    "clause": r["param"],
                    "answer": "not_needed",
                    "by": "lead@example.com",
                    "date": "2026-10-01",
                    "reason": "does not apply to this research team",
                }
            )
        elif r["status"] == "GAP" and not r["rule_ids"][0].startswith("AS-1"):
            waivers.append({"rule_id": r["param"], "reason": "accepted for the demo", "approved_by": "lead"})
    data["confirmed"] = list({c["clause"]: c for c in confirmed}.values())
    data["waivers"] = list({w["rule_id"]: w for w in waivers}.values())
    spec.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def run_path(fw: str, root: Path) -> dict[str, tuple[bool, str]]:
    """The whole path for one framework; each assertion's result and, when it fails, why."""
    d = root / fw
    shutil.copytree(MATRIX / fw, d)
    (d / "extra_tools.json").write_text("[]", encoding="utf-8")
    cwd = os.getcwd()
    os.chdir(d)
    res: dict[str, tuple[bool, str]] = {}
    outputs: list[str] = []
    try:
        outputs.append(_invoke(["audit", "--quiet"]).output)
        _invoke(["init", "--quiet"])
        text = (d / "spec.yml").read_text(encoding="utf-8")
        obs = build_observed(d)
        names = [*obs.declared_tools, *code_agents(obs)]
        body = text.split("\nagent_id:")[1]
        leaked = [
            n
            for n in names
            if f'"{n}"' in body
            or re.search(
                rf"^#.*\b{re.escape(n)}\b.*(role_boundary|"
                rf"agent_registry|declared_delegation)",
                body,
                re.M,
            )
        ]
        res[ASSERTIONS[0]] = (
            not leaked and "role_boundary:     TODO" in text and "agent_registry: TODO" in text,
            f"names in the spec: {leaked}",
        )
        _fill(d / "spec.yml", d)
        first = _json(["check"])
        assert (d / "spec.yml").is_file()
        _invoke(["mocks", "--quiet"])
        _mock_run(d)
        after = _json(["check"])
        outputs.append(_invoke(["check", "--all", "--verbose", "--quiet"], 60).output)
        outputs.append(_invoke(["check", "--all", "--quiet"], 120).output)
        _confirm(d / "spec.yml", after)
        final = _json(["check"])
        final_out = _invoke(["check"]).output
        outputs.append(final_out)
        coverage = _invoke(["check", "--coverage"]).stdout
        reports = [first, after, final]

        clash = []
        for rep in reports:
            never = {canonical(r["param"]) for r in rep["results"] if r["rule_ids"] == ["AS-201"]}
            offered = (
                {
                    canonical(t)
                    for c in json.loads((d / ".agenticreq" / "captures.json").read_text())["calls"]
                    for t in (x["name"] for x in c["tools"])
                }
                if rep is not first
                else set()
            )
            clash += sorted(never & offered)
            keys = [(r["agent"], r["param"]) for r in rep["results"]]
            clash += [f"{k} twice" for k in keys if keys.count(k) > 1]
        res[ASSERTIONS[1]] = (not clash, f"in two lists: {clash}")

        ids = [
            r["reason"]
            for rep in reports
            for r in rep["results"]
            if r["status"] == "GAP"
            and any(looks_generated(w) and len(w) >= 6 for w in re.findall(r"\b[0-9a-f]{6,}\b", r["reason"]))
        ]
        res[ASSERTIONS[2]] = (not ids, f"{ids}")

        again = [r["reason"] for r in after["results"] if RUN in r["reason"] and "mocks.yml" not in r["reason"]]
        again += [r["reason"] for r in after["results"] if r["status"] == "CHECK" and "mocks.yml" not in r["reason"]]
        res[ASSERTIONS[3]] = (not again, f"{again}")

        written = [p for p in after["results"] if p["param"] in set(sign_off_only())]
        from agenticreq.loader import load_spec
        from agenticreq.subfields import written as written_paths

        spec_obj = load_spec(d / "spec.yml").spec
        missing = [
            p
            for p in written_paths(spec_obj)
            if not rule_of(p)  # type: ignore[arg-type]
            and not any(r["param"] == p for r in final["results"])
        ]
        res[ASSERTIONS[4]] = (not missing, f"no rule and not listed: {missing}; listed {len(written)}")

        nn = {r["param"]: r["status"] for r in first["results"] if r["status"] == "NOT NEEDED"}
        want = {
            "approval_required",
            "escalation",
            "data_rules",
            "conflict_resolution",
            "goal_drift_guard",
            "tool_schema_strictness",
            "bounds.max_cost_usd",
        }
        res[ASSERTIONS[5]] = (want <= set(nn), f"missing NOT NEEDED: {sorted(want - set(nn))}")

        cut = [ln for o in outputs for ln in o.splitlines() if "…" in ln or "..." in ln]
        listed = [
            ln.split()[0]
            for ln in coverage.splitlines()[2:]
            if ln.strip() and not ln[0].isdigit() and not ln.startswith("sub-fields")
        ]
        complete = listed == [p.name for p in PARAMS if p.level <= 3]
        res[ASSERTIONS[6]] = (not cut and complete, f"cut: {cut[:2]}; coverage complete: {complete}")

        res[ASSERTIONS[7]] = (
            ALL_DONE in final_out and final["counts"]["gaps"] == 0,
            f"counts {final['counts']}; "
            + " | ".join(
                f"{r['status']} {r['param']}: {r['reason']}"
                for r in final["results"]
                if r["status"] in ("GAP", "SIGN-OFF", "CHECK")
            )[:600],
        )
    finally:
        os.chdir(cwd)
    return res


@pytest.fixture(autouse=True)
def _state_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTICREQ_NO_STATE", raising=False)  # the progress block reads the state file


@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_the_whole_path(fw: str, tmp_path: Path) -> None:
    res = run_path(fw, tmp_path)
    failed = {k: why for k, (ok, why) in res.items() if not ok}
    assert not failed, failed
    assert list(res) == ASSERTIONS


def table(cell: Callable[[bool], str] = lambda ok: "pass" if ok else "FAIL") -> str:
    os.environ.pop("AGENTICREQ_NO_STATE", None)
    lines = [
        "| framework | " + " | ".join(a.split(" ", 1)[0] for a in ASSERTIONS) + " |",
        "|---" * (len(ASSERTIONS) + 1) + "|",
    ]
    notes = []
    with tempfile.TemporaryDirectory() as tmp:
        for fw in FRAMEWORKS:
            res = run_path(fw, Path(tmp))
            lines.append(f"| {fw} | " + " | ".join(cell(res[a][0]) for a in ASSERTIONS) + " |")
            notes += [f"{fw} / {a}: {why}" for a, (ok, why) in res.items() if not ok]
    return "\n".join(lines + [""] + [f"{i + 1}. {a}" for i, a in enumerate(ASSERTIONS)] + notes)


if __name__ == "__main__":
    print(table())
