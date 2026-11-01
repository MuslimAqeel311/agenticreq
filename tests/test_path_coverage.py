"""P0 / P0b / P1 from the scenario_crewai_hard run.

- an approval gate PASSes only when it runs on every path to the tool's action (a bypassable gate is UNKNOWN);
- `proven` never labels a claim that something is absent from the code;
- tools built at runtime are extracted, and the ones that cannot be resolved are reported, not dropped.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from conftest import chat_call

from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.models.finding import Finding
from agenticreq.observe import build_observed

SPEC = """
spec_version: 1
agent_id: po
risk_level: high
goal: {statement: buy yarn, done_criteria: [po submitted]}
out_of_scope: [paying invoices]
role_boundary: {allowed_tools: [submit_purchase_order]}
prohibited: [{tool: delete_db, reason: never}]
approval_required: [{tool: submit_purchase_order, approver: head, timeout: 1h, on_timeout: safe_stop}]
limits: []
escalation: []
untrusted_inputs: []
owner: a@b.c
technical_owner: a@b.c
data_rules: {}
"""

CREW_TOOL = """
from crewai.tools import BaseTool
from crewai import Agent

def require_approval(tool, payload):
    return False

class SubmitPurchaseOrderTool(BaseTool):
    name: str = "submit_purchase_order"
    description: str = "Submit a purchase order."

    def _post(self, payload):
        STORE[payload["po_id"]] = payload

    def _run(self, po_id: str, urgent: bool = False) -> str:
        payload = {{"po_id": po_id}}
{body}
        return "submitted"

agent = Agent(role="buyer", goal="g", backstory="b", tools=[SubmitPurchaseOrderTool()], llm="gpt-4o-2024-08-06")
"""

# the scenario's P5, verbatim shape: approval only when not urgent
BYPASS = """\
        if not urgent:
            if not require_approval("submit_purchase_order", payload):
                return "rejected"
        self._post(payload)"""


def _check(tmp: Path, src: str, spec: str | None = SPEC, mode: str = "check") -> tuple[list[Finding], set[str]]:
    (tmp / "agent.py").write_text(textwrap.dedent(src), encoding="utf-8")
    loaded = load_spec_text(textwrap.dedent(spec)) if spec else None
    r = run(mode=mode if loaded else "audit", loaded=loaded, observed=build_observed(tmp))
    return r.findings, {f"{p.rule_id}:{p.subject}" for p in r.passed}


def _as205(fs: list[Finding]) -> list[Finding]:
    return [f for f in fs if f.rule_id == "AS-205"]


# ------------------------------------------------------------------ P0: the false PASS


def test_bypassable_approval_never_passes(tmp_path: Path) -> None:
    fs, passed = _check(tmp_path, CREW_TOOL.format(body=BYPASS))
    assert "AS-205:submit_purchase_order" not in passed
    [f] = _as205(fs)
    assert f.confidence.level.value == "unknown"
    assert "if not urgent" in f.detail and "not verified" in f.title


@pytest.mark.parametrize(
    "body,verified",
    [
        # gate first, decision honoured: every path to _post passes it
        (
            '        if not require_approval("submit_purchase_order", payload):\n            return "rejected"\n'
            "        self._post(payload)",
            True,
        ),
        (
            '        ok = require_approval("submit_purchase_order", payload)\n        if not ok:\n            return "no"\n'
            "        self._post(payload)",
            True,
        ),
        # an early branch already does the protected write
        (
            '        if urgent:\n            self._post(payload)\n            return "fast"\n'
            '        if not require_approval("submit_purchase_order", payload):\n            return "rejected"\n'
            "        self._post(payload)",
            False,
        ),
        # short-circuit skips the gate
        (
            '        if urgent or require_approval("submit_purchase_order", payload):\n            self._post(payload)',
            False,
        ),
        # a handler can carry on without approval
        (
            '        try:\n            require_approval("submit_purchase_order", payload)\n        except Exception:\n'
            "            pass\n        self._post(payload)",
            False,
        ),
        # loop body may run zero times
        (
            '        for _ in range(int(urgent)):\n            require_approval("submit_purchase_order", payload)\n'
            "        self._post(payload)",
            False,
        ),
    ],
)
def test_gate_passes_only_on_every_path(tmp_path: Path, body: str, verified: bool) -> None:
    fs, passed = _check(tmp_path, CREW_TOOL.format(body=body))
    assert ("AS-205:submit_purchase_order" in passed) is verified
    if not verified:
        [f] = _as205(fs)
        assert f.confidence.level.value == "unknown"


LG_TOOL = """
from langchain_core.tools import tool
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI

@tool
def submit_purchase_order(po_id: str) -> str:
    '''Submit a PO.'''
{body}
    STORE[po_id] = "submitted"
    return "ok"

llm = ChatOpenAI(model="gpt-4o-2024-08-06").bind_tools([submit_purchase_order])
app.invoke({{}}, config={{"recursion_limit": 10}})
"""


@pytest.mark.parametrize(
    "body,verified",
    [
        ('    if not interrupt({"po": po_id}):\n        return "denied"', True),
        ('    interrupt({"po": po_id})', False),  # the human's answer is thrown away
    ],
)
def test_interrupt_decision_must_be_used(tmp_path: Path, body: str, verified: bool) -> None:
    fs, passed = _check(tmp_path, LG_TOOL.format(body=body))
    assert ("AS-205:submit_purchase_order" in passed) is verified


def test_global_gate_is_unknown_not_pass(tmp_path: Path) -> None:
    src = LG_TOOL.format(body="    pass") + "\ndef review(state):\n    return interrupt('ok?')\n"
    fs, passed = _check(tmp_path, src)
    assert "AS-205:submit_purchase_order" not in passed
    [f] = _as205(fs)
    assert f.confidence.level.value == "unknown"


def test_audit_reports_partial_gate_on_a_write(tmp_path: Path) -> None:
    fs, _ = _check(tmp_path, CREW_TOOL.format(body=BYPASS), spec=None)
    f = next(f for f in fs if f.rule_id == "AS-212")
    assert "some paths" in f.title and "if not urgent" in f.detail


def test_partial_gate_is_still_a_taint_sink(tmp_path: Path) -> None:
    """A gate that can be bypassed does not take the tool out of the untrusted -> write analysis."""
    from agenticreq.ast.extractor import extract

    (tmp_path / "agent.py").write_text(textwrap.dedent(CREW_TOOL.format(body=BYPASS)), encoding="utf-8")
    g = next(g for g in extract(tmp_path).gates if g.tools == ["submit_purchase_order"])
    assert g.coverage == "partial" and not g.verified


# ------------------------------------------------------------------ P0b: absence is never `proven`


WIRE_ONLY = """
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def lookup(q: str) -> str:
    '''Look.'''
    return q

llm = ChatOpenAI(model="gpt-4o-2024-08-06").bind_tools([lookup])
app.invoke({}, config={"recursion_limit": 10})
"""


def _wire(names: list[str]) -> list:
    return [chat_call(0, [{"name": n, "parameters": {"type": "object", "properties": {}}} for n in names])]


def test_undeclared_wire_tool_is_not_proven(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(textwrap.dedent(WIRE_ONLY), encoding="utf-8")
    r = run(mode="audit", observed=build_observed(tmp_path, captures=_wire(["lookup", "mystery"])))
    f = next(f for f in r.findings if f.rule_id == "AS-202")
    assert f.confidence.level.value == "heuristic"


def test_undeclared_wire_tool_is_unknown_when_tools_are_unresolved(tmp_path: Path) -> None:
    src = (
        WIRE_ONLY + "\nimport json\nSPECS = json.load(open('t.json'))\nBUILT = [tool(n)(f) for n, f in SPECS.items()]\n"
    )
    (tmp_path / "agent.py").write_text(textwrap.dedent(src), encoding="utf-8")
    r = run(mode="audit", observed=build_observed(tmp_path, captures=_wire(["lookup", "mystery"])))
    f = next(f for f in r.findings if f.rule_id == "AS-202")
    assert f.confidence.level.value == "unknown" and "could not be resolved" in f.detail


def test_limit_param_dropped_on_the_wire_is_not_missing_from_code(tmp_path: Path) -> None:
    """The scenario's AS-217 'has no parameter score': the code has it; only the wire lost it."""
    src = (
        WIRE_ONLY.replace("def lookup(q: str)", "def rate(supplier: str, score: int)")
        .replace("return q", "return supplier")
        .replace("[lookup]", "[rate]")
    )
    spec = (
        SPEC.replace(
            "approval_required: [{tool: submit_purchase_order, approver: head, timeout: 1h, on_timeout: safe_stop}]",
            "approval_required: []",
        )
        .replace("limits: []", "limits: [{tool: rate, arg: score, limit: {max: 5}}]")
        .replace("allowed_tools: [submit_purchase_order]", "allowed_tools: [rate]")
    )
    (tmp_path / "agent.py").write_text(textwrap.dedent(src), encoding="utf-8")
    wire = [chat_call(0, [{"name": "rate", "parameters": {"type": "object", "properties": {}}}])]
    r = run(
        mode="check", loaded=load_spec_text(textwrap.dedent(spec)), observed=build_observed(tmp_path, captures=wire)
    )
    assert not any(f.rule_id == "AS-217" for f in r.findings)


# ------------------------------------------------------------------ P1: tools built at runtime


@pytest.mark.parametrize(
    "builder",
    [
        # the scenario's style 5
        "def build():\n    built = []\n    for name, fn in SPECS.items():\n        built.append(tool(name)(fn))\n    return built\n",
        "def build():\n    return [tool(n)(f) for n, f in SPECS.items()]\n",
        "def build():\n    out = []\n    for name, fn in PAIRS:\n        out.append(tool(name)(fn))\n    return out\n",
    ],
)
def test_loop_built_tools_are_extracted(tmp_path: Path, builder: str) -> None:
    src = (
        "from crewai.tools import tool\nfrom crewai import Agent\n"
        "def _rate(currency: str) -> float:\n    '''Rate.'''\n    return 1.0\n"
        "def _sql(query: str) -> str:\n    '''Run SQL.'''\n    return ''\n"
        "SPECS = {'get_exchange_rate': _rate, 'run_sql': _sql}\n"
        "PAIRS = [('get_exchange_rate', _rate), ('run_sql', _sql)]\n"
        + builder
        + "agent = Agent(role='r', goal='g', backstory='b', tools=[*build()], llm='gpt-4o-2024-08-06')\n"
    )
    (tmp_path / "agent.py").write_text(src, encoding="utf-8")
    obs = build_observed(tmp_path)
    assert {"get_exchange_rate", "run_sql"} <= set(obs.declared_tools)
    assert [p.name for p in obs.declared_tools["run_sql"].params] == ["query"]
    assert {"get_exchange_rate", "run_sql"} <= set(obs.bound_tools)
    assert obs.unresolved_tools == []


def test_unresolvable_runtime_tool_is_reported(tmp_path: Path) -> None:
    src = (
        "import json\nfrom crewai.tools import tool\nfrom crewai import Agent\n"
        "def build():\n    built = []\n    for name, fn in json.load(open('tools.json')).items():\n"
        "        built.append(tool(name)(fn))\n    return built\n"
        "agent = Agent(role='r', goal='g', backstory='b', tools=build(), llm='gpt-4o-2024-08-06')\n"
    )
    fs, passed = _check(tmp_path, src, spec=SPEC.replace("submit_purchase_order", "x"))
    f = next(f for f in fs if f.rule_id == "AS-902")
    assert f.confidence.level.value == "unknown" and f.code_ref is not None and f.code_ref.line == 7
    # the prohibited tool might be one of the unseen ones: absence is not proof
    assert "AS-203:delete_db" not in passed
    assert any(f.rule_id == "AS-203" and f.confidence.level.value == "unknown" for f in fs)


def test_bypassable_gate_reported_on_the_taint_path(tmp_path: Path) -> None:
    src = (
        CREW_TOOL.format(body=BYPASS)
        .replace(
            "agent = Agent(",
            "from crewai.tools import tool\n\n@tool('read_supplier_emails')\n"
            "def read_supplier_emails(box: str) -> str:\n    '''Read the inbox.'''\n    return ''\n\nagent = Agent(",
        )
        .replace("tools=[SubmitPurchaseOrderTool()]", "tools=[SubmitPurchaseOrderTool(), read_supplier_emails]")
    )
    spec = SPEC.replace(
        "allowed_tools: [submit_purchase_order]", "allowed_tools: [submit_purchase_order, read_supplier_emails]"
    ).replace("untrusted_inputs: []", "untrusted_inputs: [{source: read_supplier_emails}]")
    fs, _ = _check(tmp_path, src, spec=spec)
    f = next(f for f in fs if f.rule_id == "AS-401")
    assert f.severity == "blocker" and "can bypass its gate" in f.title and "if not urgent" in f.detail
