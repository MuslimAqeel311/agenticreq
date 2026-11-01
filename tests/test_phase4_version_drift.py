"""Phase 4 (revised criterion): the rules detect all three drift kinds — AS-602 description dropped,
AS-603 optional sent as required, AS-605 default dropped — and every fidelity finding records the
framework name and resolved version.

Schema rewriting is version-dependent, so these tests assert what agenticreq reports on *recorded*
captures of specific framework versions. They are observations of those versions, not expectations
about the framework in general.
"""

from pathlib import Path

import pytest

from agenticreq.capture.discover import load_capture_bundle
from agenticreq.engine import run
from agenticreq.observe import build_observed
from agenticreq.rules.fidelity import compare_tool

FIX = Path(__file__).parent / "fixtures"

#: spike gaps.md #7 — the 13 fields CrewAI lost in the spike, as (field, kind)
SPIKE_CREWAI_LOSSES = [
    ("submit_order.sku", "AS-602"),
    ("submit_order.quantity", "AS-602"),
    ("submit_order.priority", "AS-602"),
    ("pay_invoice.invoice_id", "AS-602"),
    ("pay_invoice.amount", "AS-602"),
    ("get_inventory.sku", "AS-602"),
    ("get_inventory.warehouse", "AS-603"),
    ("get_inventory.warehouse", "AS-602"),
    ("get_inventory.warehouse", "AS-605"),
    ("search_policy.query", "AS-602"),
    ("search_policy.max_results", "AS-603"),
    ("search_policy.max_results", "AS-602"),
    ("search_policy.max_results", "AS-605"),
]
#: observed on CrewAI 1.9.3 with the spike-shape fixture: defaults now survive
KEPT_ON_1_9_3 = {("get_inventory.warehouse", "AS-605"), ("search_policy.max_results", "AS-605")}
#: on 1.9.3 the default survives but `required` stays, so the spike's optional→required (AS-603)
#: shows up as a contradictory contract (AS-604) instead
BECAME_CONTRADICTORY = {
    ("get_inventory.warehouse", "AS-603"): ("get_inventory.warehouse", "AS-604"),
    ("search_policy.max_results", "AS-603"): ("search_policy.max_results", "AS-604"),
}


def _obs(agent: Path, capture: str):
    calls, versions = load_capture_bundle(FIX / "captures" / capture)
    return build_observed(agent, captures=calls, runtime_versions=versions)


def _lost(obs) -> set[tuple[str, str]]:
    out = set()
    for name, t in obs.declared_tools.items():
        issues, _, _ = compare_tool(t, obs.wire_tools[name])
        out |= {(f"{name}.{param}" if param else name, rid) for rid, param, _ in issues}
    return out


@pytest.fixture(scope="module")
def spike_shape():
    return _obs(FIX / "agents_crewai" / "crew_spike_shape", "crewai_spike_shape.json")


def test_spike_shape_on_crewai_1_9_3(spike_shape) -> None:
    assert spike_shape.framework_tag() == "crewai 1.9.3 (runtime)"
    lost = _lost(spike_shape)
    expected = {BECAME_CONTRADICTORY.get(x, x) for x in set(SPIKE_CREWAI_LOSSES) - KEPT_ON_1_9_3}
    assert lost == expected  # 11 of the spike's 13 still lost (2 of them now as AS-604)
    contradictory = [f for f in run(mode="audit", observed=spike_shape).findings if f.rule_id == "AS-604"]
    assert sorted(f.title for f in contradictory) == [
        "get_inventory: contradictory field contract (warehouse)",
        "search_policy: contradictory field contract (max_results)",
    ]
    assert any(
        f.evidence.endswith("wire: required=true, default='KHI' | code: optional, default='KHI'") for f in contradictory
    )


def test_every_fidelity_finding_carries_version(spike_shape) -> None:
    r = run(mode="audit", observed=spike_shape)
    fid = [f for f in r.findings if f.rule_id.startswith("AS-60")]
    assert fid and all(f.evidence.startswith("crewai 1.9.3 (runtime) | ") for f in fid)
    summary = [f for f in fid if f.rule_id == "AS-608"]
    assert summary and all("version-dependent" in f.detail for f in summary)


def test_all_three_drift_kinds_detected_on_real_frameworks(spike_shape) -> None:
    crew = {f.rule_id for f in run(mode="audit", observed=spike_shape).findings}
    smol = _obs(FIX / "agents" / "smol_tools", "smol_tools.json")
    r = run(mode="audit", observed=smol)
    smol_ids = {f.rule_id for f in r.findings}
    assert {"AS-602", "AS-604"} <= crew  # CrewAI 1.9.3 (its optional→required fields keep the default)
    # AS-603 (required, default absent — the spike's CrewAI shape) does not occur in any real capture
    # we hold; it is exercised by tests/test_rules_matrix.py only.
    assert "AS-605" in smol_ids  # smolagents 1.26.0
    f605 = next(f for f in r.findings if f.rule_id == "AS-605")
    assert f605.evidence.startswith("smolagents 1.26.0 (runtime) | ")


def test_runtime_version_beats_analyzer_install(tmp_path: Path) -> None:
    calls, _ = load_capture_bundle(FIX / "captures" / "smol_tools.json")
    obs = build_observed(FIX / "agents" / "smol_tools", captures=calls, runtime_versions={"smolagents": "9.9.9"})
    assert obs.framework_tag() == "smolagents 9.9.9 (runtime)"  # not the version installed here
    no_runtime = build_observed(FIX / "agents" / "smol_tools", captures=calls)
    import importlib.util

    # without a runtime version the analyzer's own install is used — or the tag says it is unresolved
    expected = "(installed)" if importlib.util.find_spec("smolagents") else "(version unresolved)"
    assert no_runtime.framework_tag().endswith(expected)


def test_unresolved_version_is_said_not_omitted(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("from someframework import tool\n")
    from agenticreq.models.observed import ObservedAgent

    o = ObservedAgent(framework="crewai")
    assert o.framework_tag() == "crewai (version unresolved)"


def test_other_version_sensitive_rules_tagged() -> None:
    # AS-202 allowlist note, AS-3xx delegation inferred from handoff conventions
    obs = _obs(FIX / "agents_crewai" / "crew_po", "crewai_po.json")
    r = run(mode="audit", observed=obs)
    assert any(p.rule_id == "AS-202" and "crewai 1.9.3 (runtime)" in p.note for p in r.passed)
    from agenticreq.loader import load_spec_text

    spec = load_spec_text("agent_id: m\nrisk_level: low\nagent_registry: [{id: x}]\ndeclared_delegation: []\n")
    rc = run(mode="check", loaded=spec, observed=obs)
    graph = [f for f in rc.findings if f.rule_id in ("AS-301", "AS-302", "AS-303", "AS-304", "AS-305", "AS-308")]
    assert {"AS-301", "AS-308"} <= {f.rule_id for f in graph}  # non-vacuous
    assert all(f.evidence.startswith("crewai 1.9.3 (runtime) | ") for f in graph)
    code = _obs(FIX / "agents" / "smol_code", "smol_tools.json")  # framework detected from smol_code source
    r2 = run(mode="audit", observed=code)
    assert next(f for f in r2.findings if f.rule_id == "AS-901").evidence.startswith("smolagents 1.26.0")
