"""Level-2 no-spec checks recognise the real construct in each framework (true positives), and the
parameter catalog accounts for every spec field (nothing silently unchecked)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agenticreq.engine import _load_rule_modules, run
from agenticreq.loader import load_spec_text
from agenticreq.observe import build_observed

SATISFIED = {
    "crewai": """
        import os
        from crewai import Agent, Crew, Task
        from crewai.tools import tool

        @tool("lookup")
        def lookup(q: str) -> str:
            '''Look something up.'''
            return q

        budget = BudgetManager(max_budget=2.0)
        a = Agent(role="r", goal="g", backstory="b", tools=[lookup], llm=os.getenv("MODEL"),
                  max_iter=5, max_execution_time=120, step_callback=budget.check)
        t = Task(description="d", expected_output="e", agent=a, output_pydantic=Report)
        crew = Crew(agents=[a], tasks=[t], max_rpm=10)
        crew.kickoff()
    """,
    "pydantic-ai": """
        import asyncio
        from pydantic_ai import Agent
        from pydantic_ai.usage import UsageLimits
        from pydantic_settings import BaseSettings
        import logfire

        logfire.instrument_pydantic_ai()  # an import alone traces nothing

        class Settings(BaseSettings):
            model: str = "openai:gpt-4o-2024-08-06"

        agent = Agent("openai:gpt-4o-2024-08-06", output_type=Answer)

        @agent.tool_plain
        def lookup(q: str) -> str:
            '''Look something up.'''
            return q

        async def main():
            return await asyncio.wait_for(agent.run("x", usage_limits=UsageLimits(request_limit=5,
                                                     total_tokens_limit=20000)), 60)
    """,
    "langgraph": """
        import os
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver

        @tool
        def lookup(q: str) -> str:
            '''Look something up.'''
            return q

        graph = create_react_agent(os.environ["MODEL"], tools=[lookup], checkpointer=MemorySaver(),
                                   response_format=Answer)
        graph.invoke({}, config={"recursion_limit": 10, "callbacks": [handler]}, max_budget_usd=1.0,
                     timeout=60)
    """,
}


def _audit(tmp: Path, src: str, spec: str | None = None) -> tuple[set[str], set[str]]:
    (tmp / "agent.py").write_text(textwrap.dedent(src), encoding="utf-8")
    obs = build_observed(tmp)
    r = run(mode="check" if spec else "audit", loaded=load_spec_text(spec) if spec else None, observed=obs)
    return {f.rule_id for f in r.findings}, {p.rule_id for p in r.passed}


@pytest.mark.parametrize("fw", sorted(SATISFIED))
def test_level2_constructs_recognised(tmp_path: Path, fw: str) -> None:
    got, passed = _audit(tmp_path, SATISFIED[fw])
    for rid in ("AS-221", "AS-222", "AS-223", "AS-224", "AS-225"):
        assert rid not in got, f"{fw}: {rid} fired although the construct is present"
        assert rid in passed, f"{fw}: {rid} neither fired nor passed"


@pytest.mark.parametrize(
    "body,fires",
    [
        ("try:\n        y = 1\n    except:\n        pass\n    return 'ok'", True),
        ("try:\n        y = 1\n    except Exception:\n        return None\n    return 'ok'", True),
        ("try:\n        y = 1\n    except KeyError:\n        pass\n    return 'ok'", False),  # narrow: deliberate
        ("try:\n        y = 1\n    except Exception:\n        raise\n    return 'ok'", False),
        ("try:\n        y = 1\n    except Exception as e:\n        return str(e)\n    return 'ok'", False),
        ("try:\n        y = 1\n    except Exception:\n        return 'Error: lookup failed'\n    return 'ok'", False),
        # logging tells an operator, not the agent: the agent still receives 'ok'
        ("try:\n        y = 1\n    except Exception:\n        log.exception('x')\n    return 'ok'", True),
        # real-repo shapes (validation run 3)
        ("try:\n        y = 1\n    except Exception as e:\n        print(f'Error: {e}')\n        return ''", True),
        (
            "try:\n        y = 1\n    except Exception as e:\n        if is_limit(e) or True:\n"
            "            return Command(goto=END)\n    return 'ok'",
            True,
        ),
        (
            "try:\n        y = 1\n    except Exception as e:\n        log.warning(e)\n        return fallback_text",
            False,
        ),
        ("try:\n        y = 1\n    except Exception as e:\n        state['error'] = str(e)\n    return 'ok'", False),
        ("try:\n        y = 1\n    except Exception as e:\n        errors.append(e)\n    return 'ok'", False),
    ],
)
def test_fail_open_detection(tmp_path: Path, body: str, fires: bool) -> None:
    src = f"from langchain_core.tools import tool\n@tool\ndef lookup(q: str) -> str:\n    '''Look.'''\n    {body}\n"
    got, _ = _audit(tmp_path, src)
    assert ("AS-219" in got) is fires


@pytest.mark.parametrize(
    "name,body,sev",
    [
        ("update_system_prompt", "agent.instructions = q", "blocker"),
        ("rewrite_file", "Path(__file__).write_text(q)", "blocker"),
        ("add_tool", "return q", "warning"),  # name only: weaker evidence, weaker severity
    ],
)
def test_self_modification(tmp_path: Path, name: str, body: str, sev: str) -> None:
    src = f"from langchain_core.tools import tool\n@tool\ndef {name}(q: str) -> str:\n    '''Do.'''\n    {body}\n"
    (tmp_path / "agent.py").write_text(src, encoding="utf-8")
    f = next(f for f in run(mode="audit", observed=build_observed(tmp_path)).findings if f.rule_id == "AS-220")
    assert f.severity == sev


def test_business_tool_named_policy_is_not_self_modification(tmp_path: Path) -> None:
    src = "from langchain_core.tools import tool\n@tool\ndef update_policy(q: str) -> str:\n    '''Insurance.'''\n    return q\n"
    got, _ = _audit(tmp_path, src)
    assert "AS-220" not in got


def test_prod_url_literal_flagged(tmp_path: Path) -> None:
    src = (
        "import os\nfrom langchain_core.tools import tool\nBASE = 'https://api.prod.example.com/v1'\n"
        "@tool\ndef lookup(q: str) -> str:\n    '''Look.'''\n    return os.getenv('X') or q\n"
    )
    (tmp_path / "agent.py").write_text(src, encoding="utf-8")
    fs = [f for f in run(mode="audit", observed=build_observed(tmp_path)).findings if f.rule_id == "AS-223"]
    assert [f.title for f in fs] == ["Production endpoint is hard-coded"]


def test_per_request_timeout_is_not_a_run_bound(tmp_path: Path) -> None:
    src = (
        "import requests\nfrom langchain_core.tools import tool\n@tool\ndef fetch_page(q: str) -> str:\n"
        "    '''Fetch.'''\n    return requests.get('https://x.org', timeout=5).text\n"
    )
    got, _ = _audit(tmp_path, src)
    assert "AS-222" in got


def test_cross_agent_trust_none_makes_sibling_output_untrusted(tmp_path: Path) -> None:
    src = textwrap.dedent("""
        import requests
        from agents import Agent, function_tool

        @function_tool
        def issue_refund(order_id: str, idempotency_key: str, dry_run: bool = True) -> str:
            '''Refund an order.'''
            if dry_run:
                return order_id
            requests.post("https://pay.example/refunds", json={"id": order_id, "key": idempotency_key})  # writes
            return order_id

        billing = Agent(name="billing", instructions="b", tools=[issue_refund])
        triage = Agent(name="triage", instructions="t", handoffs=[billing])
    """)
    base = textwrap.dedent("""
        agent_id: billing
        risk_level: {risk}
        goal: {{statement: g, done_criteria: [d]}}
        out_of_scope: []
        prohibited: []
        approval_required: []
        limits: []
        escalation: []
        untrusted_inputs: []
        owner: o
        technical_owner: t
        topology: supervisor
        agent_registry: [{{id: triage}}, {{id: billing}}]
        declared_delegation: [{{from: triage, to: billing}}]
        {extra}
    """)
    (tmp_path / "agent.py").write_text(src, encoding="utf-8")
    obs = build_observed(tmp_path)
    zero = run(
        mode="check", loaded=load_spec_text(base.format(risk="medium", extra="cross_agent_trust: none")), observed=obs
    )
    f = next(f for f in zero.findings if f.rule_id == "AS-401")
    assert f.path and f.path[0] == "triage" and f.path[-1] == "issue_refund"
    verified = run(mode="check", loaded=load_spec_text(base.format(risk="medium", extra="")), observed=obs)
    assert not any(f.rule_id == "AS-401" for f in verified.findings)


# ------------------------------------------------------------------ the parameter catalog
def test_catalog_covers_every_spec_field() -> None:
    from agenticreq.models.spec import TIER1_FIELDS, TIER2_FIELDS, TIER3_FIELDS
    from agenticreq.params import level

    assert {p.name for p in level(1)} == set(TIER1_FIELDS) | {"technical_owner"}
    assert {p.name for p in level(2)} == set(TIER2_FIELDS)
    assert {p.name for p in level(3)} == set(TIER3_FIELDS)
    assert {p.name for p in level(4)} == {"trace_coverage", "enforceability"}


def test_catalog_rules_exist_and_attestations_have_reasons() -> None:
    from agenticreq.params import PARAMS
    from agenticreq.rules.attest import ATTESTATION_CLAUSES
    from agenticreq.rules.registry import REGISTRY

    _load_rule_modules()
    for p in PARAMS:
        for rid in p.rules:
            assert rid in REGISTRY, f"{p.name}: {rid} not registered"
        if p.how in ("attestation", "runtime"):  # runtime clauses fall back to attestation without real runs
            assert p.name in ATTESTATION_CLAUSES, f"{p.name} is verified by attestation but has no reason"
        if p.how == "runtime":
            assert p.rules and p.rules[0] in REGISTRY, f"{p.name}: runtime-checked but names no rule"
        if p.how == "code":
            assert p.rules, f"{p.name}: verified in code but names no rule"


def test_audit_checkable_params_have_audit_rules() -> None:
    from agenticreq.params import NEEDS_SPEC, audit_checkable
    from agenticreq.rules.base import implementations

    _load_rule_modules()
    audit_ids = {rid for impl in implementations("audit") for rid in impl.rule_ids}
    for p in audit_checkable():
        assert set(p.rules) & audit_ids, f"{p.name} is advertised as audit-checkable but no rule runs in audit"
        assert p.level == 2
    assert not set(NEEDS_SPEC) & {p.name for p in audit_checkable()}


def test_init_scaffold_has_a_todo_for_every_level1_field(tmp_path: Path) -> None:
    import yaml

    from agenticreq.params import level
    from agenticreq.scaffold import render_init

    data = yaml.safe_load(render_init(build_observed(tmp_path)))
    for p in level(1):
        assert data.get(p.name) == "TODO", p.name


# ------------------------------------------------------------------ AS-216: egress guards written in code
_FETCH = (
    "import re, requests\nfrom urllib.parse import urlparse\nfrom langchain_core.tools import tool\n"
    "ALLOWED_HOSTS = {'api.example.com'}\n@tool\ndef fetch_url(url: str) -> str:\n    '''Fetch.'''\n{body}\n"
    "    return requests.get(url, timeout=5).text\n"
)


@pytest.mark.parametrize(
    "body,guarded",
    [
        ("    if urlparse(url).hostname not in ALLOWED_HOSTS:\n        raise ValueError('host')", True),
        ("    host = urlparse(url).netloc\n    if host not in ALLOWED_HOSTS:\n        return 'refused'", True),
        ("    assert urlparse(url).hostname in {'api.example.com', 'docs.example.com'}", True),
        ("    if not url.startswith(('https://api.example.com/',)):\n        raise PermissionError(url)", True),
        ("    if not re.fullmatch(r'https://api\\.example\\.com/.*', url):\n        raise ValueError(url)", True),
        ("    validate_url(url)", True),
        ("    if not is_allowed(url):\n        return 'blocked'", True),
        # look-alikes that restrict nothing
        ("    pass", False),
        ("    if url in cache:\n        return cache[url]", False),  # a cache, not an allowlist
        ("    if not url.startswith('http'):\n        return 'bad scheme'", False),  # scheme only, any host
        ("    if urlparse(url).hostname not in ALLOWED_HOSTS:\n        print('warning')", False),  # does not stop
    ],
)
def test_egress_guard_recognised(tmp_path: Path, body: str, guarded: bool) -> None:
    got, passed = _audit(tmp_path, _FETCH.replace("{body}", body))
    assert ("AS-216" in got) is not guarded
    assert ("AS-216" in passed) is guarded


# ------------------------------------------------------------------ AS-217: prohibited tool spelled differently
_SPEC = textwrap.dedent("""
    agent_id: a
    risk_level: medium
    goal: {{statement: g, done_criteria: [d]}}
    out_of_scope: []
    prohibited: [{{tool: {tool}, reason: finance only}}]
    approval_required: []
    limits: []
    escalation: []
    untrusted_inputs: []
    owner: o
    technical_owner: t
""")
_PAY = (
    "from langchain_core.tools import tool\n@tool\ndef payInvoice(invoice_id: str) -> str:\n"
    "    '''Pay.'''\n    return invoice_id\n@tool\ndef get_inventory(sku: str) -> str:\n    '''Stock.'''\n"
    "    return sku\n"
)


@pytest.mark.parametrize(
    "prohibited,fires",
    [
        ("pay_invoice", True),  # code says payInvoice: the prohibition misses it
        ("pay_invoices", True),
        ("delete_database", False),  # absent and nothing similar: the prohibition holds
    ],
)
def test_prohibited_near_miss(tmp_path: Path, prohibited: str, fires: bool) -> None:
    (tmp_path / "agent.py").write_text(_PAY, encoding="utf-8")
    r = run(mode="check", loaded=load_spec_text(_SPEC.format(tool=prohibited)), observed=build_observed(tmp_path))
    hits = [f for f in r.findings if f.rule_id == "AS-217"]
    assert bool(hits) is fires
    passed = any(p.rule_id == "AS-203" and p.subject == prohibited for p in r.passed)
    assert passed is not fires  # a near miss is never a PASS
    if fires:
        assert "payInvoice" in hits[0].title
    from agenticreq.derived import enforceability

    assert r.metadata is not None
    row = next(
        x
        for x in enforceability(
            load_spec_text(_SPEC.format(tool=prohibited)).spec,  # type: ignore[arg-type]
            build_observed(tmp_path),
            r.passed,
        )["rows"]
        if x["clause"].startswith("prohibited")
    )
    assert bool(row["code"]) is not fires
