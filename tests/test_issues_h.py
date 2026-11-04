"""Priority H: is a model id a fixed version? Syntax decides, or a person does."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from agenticreq.ast.classify import is_pinned, pin_judge
from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.models.verdict import results
from agenticreq.observe import build_observed
from test_issues_b import ISSUES

SPEC = (ISSUES / "h1_models" / "spec.yml").read_text(encoding="utf-8")


def _line(code: str, tmp_path: Path, extra: str = "") -> object:
    (tmp_path / "a.py").write_text(
        f"from agents import Agent\nx = Agent(name='a', instructions='i', model={code})\n", encoding="utf-8"
    )
    report = run(
        mode="check", loaded=load_spec_text(SPEC + extra), observed=build_observed(tmp_path), today=date(2026, 11, 4)
    )
    return {ln.param: ln for ln in results(report)}["version_pinning"]


@pytest.mark.parametrize(
    "model,state,rule",
    [
        ("gpt-4o-2024-08-06", "pinned", "dated suffix -YYYY-MM-DD"),
        ("claude-sonnet-4-5-20250929", "pinned", "dated suffix -YYYYMMDD"),
        ("gemini-2.0-flash-001", "pinned", "numeric version suffix"),
        ("gpt-4-0613", "pinned", "numeric version suffix"),
        ("claude-3-5-sonnet-latest", "floating", "-latest marker"),
        ("gemini-2.5-flash-preview-05-20", "floating", "-preview marker"),
        ("gemini-exp-1206", "floating", "-exp marker"),
        ("gpt-4o", "floating", "bare family name"),
        ("gemini-2.0-flash", "undecidable", "no version marker"),
        ("llama-3.3-70b-versatile", "undecidable", "no version marker"),
    ],
)
def test_h1_syntactic_rules(model: str, state: str, rule: str) -> None:
    assert pin_judge(model) == (state, rule)


@pytest.mark.parametrize("prefix", ["openai/", "anthropic/", "vertex_ai/", "bedrock/", "groq/", "gemini/"])
def test_h2_routing_prefixes_are_removed(prefix: str) -> None:
    assert pin_judge(prefix + "some-model-2024-08-06")[0] == "pinned"


@pytest.mark.parametrize("model", ["azure/my-prod-deployment", "ollama/llama3:latest"])
def test_h2_deployments_and_local_tags_are_for_a_person(model: str) -> None:
    assert pin_judge(model)[0] == "undecidable" and not is_pinned(model)


def test_h1_an_undecidable_id_is_a_question_never_not_pinned(tmp_path: Path) -> None:
    ln = _line("'gemini-2.0-flash'", tmp_path)
    assert ln.status == "SIGN-OFF" and "Is this model id a fixed version?" in ln.reason  # type: ignore[attr-defined]
    assert "not pinned" not in ln.reason.lower() and "a.py:2" in ln.reason  # type: ignore[attr-defined]


def test_h1_a_floating_id_stays_a_gap_with_its_rule_and_place(tmp_path: Path) -> None:
    ln = _line("'gpt-4o'", tmp_path)
    assert ln.status == "GAP" and "bare family name" in ln.reason and "a.py:2" in ln.reason  # type: ignore[attr-defined]


def test_h1_a_pinned_id_prints_its_rule_and_place(tmp_path: Path) -> None:
    ln = _line("'gpt-4o-2024-08-06'", tmp_path)
    assert ln.status == "PASS" and "dated suffix" in ln.reason and "a.py:2" in ln.reason  # type: ignore[attr-defined]


def test_h3_every_source_of_a_model_id_is_read() -> None:
    obs = build_observed(ISSUES / "h1_models")
    by_name = {m.name: m for m in obs.models}
    assert {
        "gpt-4o-2024-08-06",
        "claude-sonnet-4-5-20250929",
        "gpt-4.1-mini-2025-04-14",
        "gemini-2.0-flash-001",
    } <= set(by_name)
    assert by_name["gpt-4.1-mini-2025-04-14"].assumed_from == "default in code of env NEW_MODEL"
    assert by_name["gpt-4.1-2025-04-14"].assumed_from == ".env.example of env AGENT_MODEL"
    assert [str(u.code_ref) for u in obs.unresolved_models] == ["agent.py:19"]  # only the runtime-only one


def test_h4_a_spec_id_counts_as_pinned(tmp_path: Path) -> None:
    ln = _line("'azure/my-prod-deployment'", tmp_path, "pinned_models: [azure/my-prod-deployment]\n")
    assert ln.status == "PASS" and "pinned_models" in ln.reason  # type: ignore[attr-defined]


def test_h4_a_spec_id_different_from_the_code_is_a_gap(tmp_path: Path) -> None:
    ln = _line("'gpt-4o-2024-08-06'", tmp_path, "pinned_models: [gpt-4o-2024-11-20]\n")
    assert ln.status == "GAP"  # type: ignore[attr-defined]
    assert "gpt-4o-2024-08-06" in ln.reason and "gpt-4o-2024-11-20" in ln.reason and "a.py:2" in ln.reason  # type: ignore[attr-defined]


def test_h4_a_spec_without_pinned_models_still_validates() -> None:
    assert load_spec_text(SPEC).spec is not None
