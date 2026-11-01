"""Phase 6 acceptance: delegation graph from CrewAI + OpenAI Agents captures, heuristic confidence,
exact paths (A → B → C can reach X); check = Tier 1+2+3."""

from pathlib import Path

import pytest
from conftest import chat_call

from agenticreq.capture.discover import load_captures
from agenticreq.engine import run
from agenticreq.loader import load_spec, load_spec_text
from agenticreq.models.confidence import ConfidenceLevel
from agenticreq.observe import build_observed

FIX = Path(__file__).parent / "fixtures"
CAP = FIX / "captures"


@pytest.mark.parametrize(
    "name,via", [("crewai_po", "delegate_work_to_coworker"), ("oai_agents_po", "transfer_to_policy_analyst")]
)
def test_graph_from_captures_is_heuristic(tmp_path: Path, name: str, via: str) -> None:
    obs = build_observed(tmp_path, captures=load_captures(CAP / f"{name}.json"))  # captures only
    assert len(obs.nodes) == 2  # two distinct system-prompt hashes
    assert len(obs.edges) == 1
    e = obs.edges[0]
    assert e.via_tool == via and e.confidence.level is ConfidenceLevel.HEURISTIC


def test_three_hop_path_printed() -> None:
    d = FIX / "agents" / "oai_agents_chain"
    obs = build_observed(d, captures=load_captures(CAP / "oai_agents_chain.json"))
    r = run(mode="check", loaded=load_spec(d / "spec.yml"), observed=obs)
    f = next(f for f in r.findings if f.rule_id == "AS-304")
    assert f.title == "procurement-triage → policy-analyst → payments can reach pay_invoice"
    assert f.path == ["procurement-triage", "policy-analyst", "payments"]
    assert any(x.rule_id == "AS-301" and x.path == ["policy-analyst", "payments"] for x in r.findings)
    # pay_invoice belongs to Payments, not to triage itself: no direct AS-203
    assert not any(x.rule_id == "AS-203" for x in r.findings)


def test_three_hop_heuristic_without_code(tmp_path: Path) -> None:
    spec = load_spec(FIX / "agents" / "oai_agents_chain" / "spec.yml")
    obs = build_observed(tmp_path, captures=load_captures(CAP / "oai_agents_chain.json"))
    r = run(mode="check", loaded=spec, observed=obs)
    f = next(f for f in r.findings if f.rule_id == "AS-304")
    assert f.confidence.level is ConfidenceLevel.HEURISTIC
    assert " → ".join(f.path or []) == "procurement-triage → policy-analyst → payments"


SPEC = """
agent_id: A
risk_level: medium
goal: {statement: g, done_criteria: [d]}
out_of_scope: []
prohibited: [{tool: wire, reason: r}]
approval_required: []
limits: []
escalation: []
untrusted_inputs: []
owner: o
technical_owner: t
agent_registry:
  - {id: A, match: "agent A"}
  - {id: B, match: "agent B"}
  - {id: C, match: "agent C"}
declared_delegation: [A -> B, B -> C]
max_delegation_depth: 2
aggregate_budget: {max_cost_usd: 5}
"""


def _chain(edges: list[tuple[str, str]], tools: dict[str, list[str]]) -> list:
    """Synthetic captures: each hop is a transfer_to_<dst> call followed by a request from dst."""
    calls = []
    i = 0
    for src, dst in edges:
        t_src = [{"name": n, "parameters": {"type": "object", "properties": {}}} for n in tools.get(src, [])]
        t_src.append({"name": f"transfer_to_{dst.lower()}", "parameters": {"type": "object", "properties": {}}})
        calls.append(
            chat_call(
                i, t_src, system=f"You are agent {src}", tool_calls=[(f"transfer_to_{dst.lower()}", {})], synthetic=True
            )
        )
        i += 1
        t_dst = [{"name": n, "parameters": {"type": "object", "properties": {}}} for n in tools.get(dst, [])]
        calls.append(chat_call(i, t_dst, system=f"You are agent {dst}", synthetic=True))
        i += 1
    return calls


def test_cycle_and_depth(tmp_path: Path) -> None:
    obs = build_observed(tmp_path, captures=_chain([("A", "B"), ("B", "C"), ("C", "A")], {}))
    r = run(mode="check", loaded=load_spec_text(SPEC), observed=obs)
    cyc = [f for f in r.findings if f.rule_id == "AS-303"]
    assert cyc and cyc[0].path and cyc[0].path[0] == cyc[0].path[-1]
    assert any(f.rule_id == "AS-301" and f.path == ["C", "A"] for f in r.findings)


def test_depth_exceeded(tmp_path: Path) -> None:
    obs = build_observed(tmp_path, captures=_chain([("A", "B"), ("B", "C")], {}))
    spec = load_spec_text(SPEC.replace("max_delegation_depth: 2", "max_delegation_depth: 1"))
    r = run(mode="check", loaded=spec, observed=obs)
    f = next(f for f in r.findings if f.rule_id == "AS-302")
    assert f.path == ["A", "B", "C"]


def test_clean_chain_passes(tmp_path: Path) -> None:
    obs = build_observed(
        tmp_path, captures=_chain([("A", "B"), ("B", "C")], {"A": ["read"], "B": ["read"], "C": ["read"]})
    )
    r = run(mode="check", loaded=load_spec_text(SPEC), observed=obs)
    assert not [f for f in r.findings if f.rule_id.startswith("AS-30") and f.rule_id != "AS-307"]
    assert any(p.rule_id == "AS-301" for p in r.passed)


def test_transitive_prohibition_synthetic(tmp_path: Path) -> None:
    obs = build_observed(tmp_path, captures=_chain([("A", "B"), ("B", "C")], {"C": ["wire"]}))
    r = run(mode="check", loaded=load_spec_text(SPEC), observed=obs)
    f = next(f for f in r.findings if f.rule_id == "AS-304")
    assert f.title == "A → B → C can reach wire"


def test_authority_escalation(tmp_path: Path) -> None:
    obs = build_observed(tmp_path, captures=_chain([("A", "B")], {"A": ["read"], "B": ["read", "delete_all"]}))
    r = run(mode="check", loaded=load_spec_text(SPEC), observed=obs)
    assert any(f.rule_id == "AS-305" and "delete_all" in f.detail for f in r.findings)
