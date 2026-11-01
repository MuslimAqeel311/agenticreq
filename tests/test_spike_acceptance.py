"""Acceptance against the spike's own 7 agents and S2 captures (tests/fixtures/spike, copied from Spike/).

Every expected number below is taken from Spike/REPORT.md (F4 schema fidelity, F5 smolagents CodeAgent,
F6 tools visible ÷ defined). Captures are the spike's recorded wire bodies, imported, not re-run.
"""

from collections import Counter
from pathlib import Path

import pytest

from agenticreq.capture.discover import load_capture_bundle, load_captures
from agenticreq.engine import run
from agenticreq.models.confidence import ConfidenceLevel
from agenticreq.observe import build_observed

SPIKE = Path(__file__).parent / "fixtures" / "spike"
AGENTS = SPIKE / "test_agents"
CAPS = SPIKE / "captures"
FOUR = {"get_inventory", "search_policy", "submit_order", "pay_invoice"}

# agent -> (framework, execution model, defined tools, tools visible at the boundary)  [REPORT.md F6]
EXPECTED = {
    "langgraph_react": ("langgraph", "schema-based", FOUR, FOUR),
    "langgraph_phased": (
        "langgraph",
        "schema-based",
        FOUR | {"issue_refund"},
        {"get_inventory", "submit_order", "pay_invoice"},
    ),
    "crewai": ("crewai", "schema-based", FOUR, FOUR),
    "pydanticai": ("pydantic-ai", "schema-based", FOUR, FOUR),
    "openai_agents": ("openai-agents", "schema-based", FOUR, FOUR),
    "smolagents": ("smolagents", "schema-based", FOUR, FOUR),
    "smolagents_code": ("smolagents", "prompt-only", FOUR, set()),
}
# agent -> fidelity findings  [REPORT.md F4: fields lost code → wire]
FIDELITY = {
    "langgraph_react": {},  # 44/44
    "langgraph_phased": {},  # 34/34
    "pydanticai": {},  # 44/44
    "crewai": {"AS-602": 9, "AS-603": 2, "AS-605": 2},  # 31/44: param desc 0/9, required 7/9, default 0/2
    "openai_agents": {"AS-604": 2},  # 42/44: required 7/9 (defaults kept -> contradictory)
    "smolagents": {"AS-605": 2},  # 42/44: default 0/2
}


def _observe(name: str):
    calls, versions = load_capture_bundle(CAPS / f"{name}_S2.json")
    return build_observed(AGENTS / f"{name}_agent.py", captures=calls, runtime_versions=versions)


def _audit(name: str):
    obs = _observe(name)
    return obs, run(mode="audit", observed=obs)


def test_s2_imports_all_24_calls() -> None:
    """REPORT.md headline: S2 call coverage 24/24."""
    total = sum(len(load_captures(p)) for p in CAPS.glob("*_S2.json"))
    assert total == 24


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_static_inventory_follows_imports(name: str) -> None:
    """Tools live in shared_tools.py and are wrapped in the agent file — AST must still find them."""
    fw, model, defined, _ = EXPECTED[name]
    obs = build_observed(AGENTS / f"{name}_agent.py")
    assert obs.framework == fw
    assert set(obs.declared_tools) == defined
    assert {t.code_ref.file for t in obs.declared_tools.values() if t.code_ref} == {"shared_tools.py"}
    assert obs.execution_model == model


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_visible_over_defined(name: str) -> None:
    _, model, defined, visible = EXPECTED[name]
    obs = _observe(name)
    real_wire = {t for t in obs.wire_tools if t in defined}
    assert real_wire == visible
    assert obs.execution_model == model


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_no_false_undeclared_tools(name: str) -> None:
    """Framework-injected tools (delegate_*, transfer_to_*, final_answer) are allowlisted; nothing else is extra."""
    _, report = _audit(name)
    assert [f.title for f in report.findings if f.rule_id == "AS-202"] == []


def test_phased_invisible_tools_are_unknown_with_reasons() -> None:
    """F6: issue_refund on an unexecuted branch, search_policy called directly from Python."""
    _, report = _audit("langgraph_phased")
    f201 = {f.title.split()[0]: f for f in report.findings if f.rule_id == "AS-201"}
    assert set(f201) == {"issue_refund", "search_policy"}
    assert all(f.confidence.level is ConfidenceLevel.UNKNOWN for f in f201.values())
    assert "branch" in f201["issue_refund"].detail
    assert "called directly from Python" in f201["search_policy"].detail


@pytest.mark.parametrize("name", sorted(FIDELITY))
def test_schema_fidelity_matches_spike(name: str) -> None:
    _, report = _audit(name)
    got = Counter(f.rule_id for f in report.findings if f.rule_id.startswith("AS-6") and f.rule_id != "AS-608")
    assert dict(got) == FIDELITY[name]


def test_code_agent_is_unknown_never_pass() -> None:
    """F5: tools only as Python signatures in the prompt -> UNKNOWN, and no tool check passes."""
    obs, report = _audit("smolagents_code")
    assert obs.execution_model == "prompt-only"
    assert any(f.rule_id == "AS-901" and f.confidence.level is ConfidenceLevel.UNKNOWN for f in report.findings)
    assert {f.title.split()[0] for f in report.findings if f.rule_id == "AS-201"} == FOUR
    assert not [p for p in report.passed if p.rule_id in ("AS-201", "AS-202")]


@pytest.mark.parametrize(
    "name,pinned",
    [
        ("langgraph_react", True),
        ("openai_agents", True),
        ("smolagents", True),
        ("crewai", False),
        ("pydanticai", False),
        ("langgraph_phased", False),
    ],
)
def test_model_pinning(name: str, pinned: bool) -> None:
    _, report = _audit(name)
    assert (not any(f.rule_id == "AS-210" for f in report.findings)) is pinned


def test_delegation_seen_for_multi_agent() -> None:
    for name, via in (("crewai", "delegate_work_to_coworker"), ("openai_agents", "transfer_to_policy_analyst")):
        obs = _observe(name)
        assert any(e.via_tool == via for e in obs.edges), name


def test_guardrail_found_statically() -> None:
    """F6: the no_huge_orders input guardrail never reaches the boundary; AST must see it."""
    obs = build_observed(AGENTS / "openai_agents_agent.py")
    assert any(g.kind == "input_guardrail" for g in obs.gates)


def test_runtime_versions_mapped_to_distributions() -> None:
    obs = _observe("openai_agents")
    assert obs.framework_version == "0.20.0" and obs.framework_version_source == "runtime"


# ---------------------------------------------------------------- check: spec vs code (Tier 1+2+3)
from agenticreq.loader import load_spec  # noqa: E402

SPEC = SPIKE / "spec.yml"
# hand-verified against the agent sources: every agent lacks an approval gate on pay_invoice (AS-205)
# and enforces neither limit in a schema (AS-207: amount max, quantity min, quantity max).
COMMON = {"AS-205": 1, "AS-207": 3}
CHECK_BLOCKERS = {
    "langgraph_react": {**COMMON, "AS-401": 2},  # search_policy and both writes on one agent
    "pydanticai": {**COMMON, "AS-401": 2},
    "smolagents": {**COMMON, "AS-401": 2},
    # the stub submit_order writes nothing (body decides); pay_invoice is still a sink: approval_required lists it
    "smolagents_code": {**COMMON, "AS-401": 1},
    "langgraph_phased": {**COMMON, "AS-203": 1, "AS-401": 2},  # issue_refund reachable; prefetch taint
    "crewai": {**COMMON, "AS-301": 1, "AS-401": 2},  # undeclared delegation; analyst result -> manager
    "openai_agents": {**COMMON, "AS-301": 1, "AS-604": 2},  # handoff goes forward only: no taint into manager
}


def _check(name: str):
    calls, versions = load_capture_bundle(CAPS / f"{name}_S2.json")
    obs = build_observed(AGENTS / f"{name}_agent.py", captures=calls, runtime_versions=versions)
    return run(mode="check", loaded=load_spec(SPEC), observed=obs)


def test_spike_spec_is_valid() -> None:
    from agenticreq.engine import validate_spec

    assert validate_spec(SPEC).findings == []


@pytest.mark.parametrize("name", sorted(CHECK_BLOCKERS))
def test_check_blockers(name: str) -> None:
    report = _check(name)
    blockers = Counter(f.rule_id for f in report.by_bucket()[next(iter(report.by_bucket()))])
    assert dict(blockers) == CHECK_BLOCKERS[name]


def test_prefetch_taint_path_is_explicit() -> None:
    f = next(f for f in _check("langgraph_phased").findings if f.rule_id == "AS-401")
    assert f.path[:3] == ["search_policy", "python", "LLM"]
    assert f.confidence.level is ConfidenceLevel.HEURISTIC


def test_delegation_taint_crosses_agents() -> None:
    f = next(f for f in _check("crewai").findings if f.rule_id == "AS-401")
    assert f.path[:3] == ["search_policy", "Policy Analyst", "Procurement Manager"]
