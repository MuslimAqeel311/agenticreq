"""One positive and one negative test per rule (engineering standard).

Each case builds a tiny agent (+ optional spec / wire capture) and asserts the rule fires
(positive) or does not (negative). Rules covered elsewhere are listed in COVERED_ELSEWHERE
and the final test asserts every registered rule is accounted for.
"""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from conftest import chat_call

from agenticreq.engine import _load_rule_modules, run
from agenticreq.loader import load_spec_text
from agenticreq.observe import build_observed
from agenticreq.rules.registry import REGISTRY

BASE_SPEC = """
agent_id: a
risk_level: {risk}
goal: {{statement: g, done_criteria: [d]}}
out_of_scope: []
prohibited: {prohibited}
approval_required: {approval}
limits: {limits}
escalation: []
untrusted_inputs: {untrusted}
owner: o
technical_owner: t
{extra}
"""


def spec(
    risk: str = "medium",
    prohibited: str = "[]",
    approval: str = "[]",
    limits: str = "[]",
    untrusted: str = "[]",
    extra: str = "",
) -> str:
    return BASE_SPEC.format(
        risk=risk, prohibited=prohibited, approval=approval, limits=limits, untrusted=untrusted, extra=extra
    )


def tool_src(
    name: str, params: str = "x: str", body: str = "return 'ok'", deco: str = "@tool", doc: str = "Do it."
) -> str:
    return f"\n{deco}\ndef {name}({params}) -> str:\n    '''{doc}'''\n    {body}\n"


HEADER = "import requests\nfrom langchain_core.tools import tool\nfrom langchain_openai import ChatOpenAI\n"
#: a tool body that really writes: the body, not the name, makes a tool a write tool
WRITES = "requests.post('https://api.example/orders', json={'x': 1})\n    return 'ok'"


def code(*tools: str, bind: list[str] | None = None, model: str = "gpt-4o-2024-08-06", extra: str = "") -> str:
    names = bind if bind is not None else [t.split("def ")[1].split("(")[0] for t in tools]
    return (
        HEADER
        + "".join(tools)
        + f"\nllm = ChatOpenAI(model={model!r}).bind_tools([{', '.join(names)}])\n"
        + "app.invoke({}, config={'recursion_limit': 10})\n"
        + extra
    )


def findings(
    tmp: Path,
    src: str,
    *,
    spec_text: str | None = None,
    wire: list[dict[str, Any]] | None = None,
    mode: str = "check",
    spans: Any = None,
) -> set[str]:
    (tmp / "agent.py").write_text(textwrap.dedent(src))
    obs = build_observed(tmp, captures=[chat_call(0, wire)] if wire is not None else None, spans=spans)
    loaded = load_spec_text(spec_text) if spec_text else None
    r = run(mode="check" if loaded else "audit", loaded=loaded, observed=obs, today=date(2026, 10, 15))
    return {f.rule_id for f in r.findings}


def W(
    name: str, props: dict[str, Any] | None = None, required: list[str] | None = None, desc: str = "Do it."
) -> dict[str, Any]:
    return {
        "name": name,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": props or {"x": {"type": "string"}},
            "required": required if required is not None else ["x"],
        },
    }


MCP_SERVER = (
    "from mcp.server.fastmcp import FastMCP\nfrom mcp.types import ToolAnnotations\nmcp = FastMCP('s')\n"
    '@mcp.tool({ann})\ndef drop_table(name: str) -> str:\n    """Drop a table."""\n    return name\n'
)

CASES: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    (
        "AS-218",  # destructive by name but declared read-only: MCP clients will not ask the user
        dict(src=MCP_SERVER.format(ann="annotations=ToolAnnotations(readOnlyHint=True)"), mode="audit"),
        dict(
            src=MCP_SERVER.format(ann="annotations=ToolAnnotations(destructiveHint=True, readOnlyHint=False)"),
            mode="audit",
        ),
    ),
    (
        "AS-203",
        dict(src=code(tool_src("issue_refund")), spec_text=spec(prohibited="[{tool: issue_refund, reason: r}]")),
        dict(src=code(tool_src("get_inventory")), spec_text=spec(prohibited="[{tool: issue_refund, reason: r}]")),
    ),
    (
        "AS-205",
        dict(src=code(tool_src("pay_invoice")), spec_text=spec(approval="[{tool: pay_invoice, approver: f}]")),
        dict(
            src=code(
                tool_src("pay_invoice", body="if not interrupt('ok'):\n        return 'denied'\n    return 'ok'")
            ).replace("from langchain_core", "from langgraph.types import interrupt\nfrom langchain_core"),
            spec_text=spec(approval="[{tool: pay_invoice, approver: f}]"),
        ),
    ),
    (
        "AS-206",
        dict(
            src=code(tool_src("get_inventory"), tool_src("delete_db")),
            spec_text=spec(extra="role_boundary: {allowed_tools: [get_inventory]}"),
        ),
        dict(
            src=code(tool_src("get_inventory")), spec_text=spec(extra="role_boundary: {allowed_tools: [get_inventory]}")
        ),
    ),
    (
        "AS-207",
        dict(src=code(tool_src("pay", "amount: float")), spec_text=spec(limits="{pay: {amount: {max: 100}}}")),
        dict(
            src=code(tool_src("pay", "amount: Annotated[float, Field(le=100)]")),
            spec_text=spec(limits="{pay: {amount: {max: 100}}}"),
        ),
    ),
    (
        "AS-208",
        dict(src=HEADER + tool_src("get_x") + "\nllm = ChatOpenAI(model='gpt-4o-2024-08-06').bind_tools([get_x])\n"),
        dict(src=code(tool_src("get_x"))),
    ),
    (
        "AS-209",  # looser than the number spec.yml states (a risk_level number is only a recommendation)
        dict(
            src=code(tool_src("get_x")).replace("'recursion_limit': 10", "'recursion_limit': 500"),
            spec_text=spec(extra="bounds: {max_steps: 50}"),
        ),
        dict(src=code(tool_src("get_x")), spec_text=spec(extra="bounds: {max_steps: 50}")),
    ),
    ("AS-210", dict(src=code(tool_src("get_x"), model="gpt-4o")), dict(src=code(tool_src("get_x")))),
    (
        "AS-211",  # needs evidence beyond the name: the body mints a fresh id per call
        dict(
            src="import uuid\n"
            + code(
                tool_src(
                    "submit_order",
                    "sku: str",
                    body="requests.post('https://api.example/o', json={'id': str(uuid.uuid4())})\n    return 'ok'",
                )
            )
        ),
        dict(
            src="import uuid\n"
            + code(
                tool_src(
                    "submit_order",
                    "sku: str, idempotency_key: str",
                    body="requests.post('https://api.example/o', json={'id': idempotency_key})\n    return 'ok'",
                )
            )
        ),
    ),
    (
        "AS-212",
        dict(src=code(tool_src("submit_order", "sku: str, idempotency_key: str", body=WRITES))),
        dict(src=code(tool_src("get_order"))),
    ),
    (
        "AS-213",
        dict(src=code(tool_src("search_orders", "query: str"))),
        dict(src=code(tool_src("search_orders", "query: str, limit: int = 10"))),
    ),
    (
        "AS-214",
        dict(src=code(tool_src("get_x")) + "OPENAI_API_KEY = 'sk-proj-abcdefghijklmnopqrstuv'\n"),
        dict(src=code(tool_src("get_x")) + "OPENAI_API_KEY = os.environ['OPENAI_API_KEY']\n"),
    ),
    ("AS-215", dict(src=code(tool_src("get_x", body="return eval(x)"))), dict(src=code(tool_src("get_x")))),
    (
        "AS-216",
        dict(src=code(tool_src("get_page", "url: str", body="return requests.get(url).text"))),
        dict(src=code(tool_src("get_page", "url: str", body="return requests.get('https://intra/x').text"))),
    ),
    (
        "AS-217",
        dict(src=code(tool_src("get_x")), spec_text=spec(approval="[{tool: pay_ghost, approver: f}]")),
        dict(src=code(tool_src("get_x")), spec_text=spec(approval="[{tool: get_x, approver: f}]")),
    ),
    (
        "AS-401",
        dict(
            src=code(tool_src("read_inbox"), tool_src("pay_invoice", body=WRITES)),
            spec_text=spec(untrusted="[read_inbox]"),
        ),
        dict(src=code(tool_src("read_inbox"), tool_src("get_invoice")), spec_text=spec(untrusted="[read_inbox]")),
    ),
    (
        "AS-403",
        dict(src=code(tool_src("read_inbox"), tool_src("get_invoice")), spec_text=spec(untrusted="[]")),
        dict(src=code(tool_src("read_inbox"), tool_src("get_invoice")), spec_text=spec(untrusted="[read_inbox]")),
    ),
    ("AS-501", dict(src=code(tool_src("get_x")), spec_text=spec()), dict(src=code(tool_src("get_x")), spec_text=None)),
    (
        "AS-502",  # answer yes needs by, date, expires and evidence
        dict(
            src=code(tool_src("get_x")),
            spec_text=spec(
                extra="confirmed: [{clause: owner, by: cto, date: 2026-01-01, "
                "expires: 2099-01-01, evidence: org chart}]"
            ),
        ),
        dict(
            src=code(tool_src("get_x")),
            spec_text=spec(extra="confirmed: [{clause: owner, by: cto, expires: 2099-01-01}]"),
        ),
    ),
    (
        "AS-503",  # answer no: the owner says it is not in place yet
        dict(
            src=code(tool_src("get_x")),
            spec_text=spec(extra="confirmed: [{clause: owner, answer: no, by: cto, date: 2026-01-01, reason: hiring}]"),
        ),
        dict(src=code(tool_src("get_x")), spec_text=spec(extra="confirmed: [{clause: owner, answer: no, by: cto}]")),
    ),
    (
        "AS-504",  # answer not_needed, on an item that may be not needed
        dict(
            src=code(tool_src("get_x")),
            spec_text=spec(
                extra="confirmed: [{clause: kill_switch, answer: not_needed, "
                "by: cto, date: 2026-01-01, reason: internal demo}]"
            ),
        ),
        dict(
            src=code(tool_src("get_x")), spec_text=spec(extra="confirmed: [{clause: kill_switch, answer: not_needed}]")
        ),
    ),
    (
        "AS-601",
        dict(src=code(tool_src("get_x")), wire=[W("get_x", desc="Something else entirely")]),
        dict(src=code(tool_src("get_x")), wire=[W("get_x")]),
    ),
    (
        "AS-602",
        dict(
            src=code(tool_src("get_x", doc="Do it.\n\n    Args:\n        x: The x.\n    ")),
            wire=[W("get_x", desc="Do it.")],
        ),
        dict(
            src=code(tool_src("get_x", doc="Do it.\n\n    Args:\n        x: The x.\n    ")),
            wire=[W("get_x", {"x": {"type": "string", "description": "The x."}})],
        ),
    ),
    (
        "AS-603",
        dict(src=code(tool_src("get_x", "x: str = 'a'")), wire=[W("get_x", {"x": {"type": "string"}})]),
        dict(
            src=code(tool_src("get_x", "x: str = 'a'")),
            wire=[W("get_x", {"x": {"type": "string", "default": "a"}}, [])],
        ),
    ),
    (
        "AS-604",  # default + required on the wire: contradictory contract
        dict(src=code(tool_src("get_x", "x: str = 'a'")), wire=[W("get_x", {"x": {"type": "string", "default": "a"}})]),
        dict(
            src=code(tool_src("get_x", "x: str = 'a'")),
            wire=[W("get_x", {"x": {"type": "string", "default": "a"}}, [])],
        ),
    ),
    (
        "AS-610",
        dict(src=code(tool_src("get_x")), wire=[W("get_x", required=[])]),
        dict(src=code(tool_src("get_x")), wire=[W("get_x")]),
    ),
    (
        "AS-605",
        dict(src=code(tool_src("get_x", "x: str = 'a'")), wire=[W("get_x", required=[])]),
        dict(
            src=code(tool_src("get_x", "x: str = 'a'")),
            wire=[W("get_x", {"x": {"type": "string", "default": "a"}}, [])],
        ),
    ),
    (
        "AS-606",
        dict(src=code(tool_src("get_x", "x: int")), wire=[W("get_x", {"x": {"type": "string"}})]),
        dict(src=code(tool_src("get_x", "x: int")), wire=[W("get_x", {"x": {"type": "integer"}})]),
    ),
    (
        "AS-607",
        dict(src=code(tool_src("get_x", "x: str, y: str")), wire=[W("get_x")]),
        dict(src=code(tool_src("get_x")), wire=[W("get_x")]),
    ),
    (
        "AS-608",
        dict(src=code(tool_src("get_x", "x: int")), wire=[W("get_x", {"x": {"type": "string"}})]),
        dict(src=code(tool_src("get_x")), wire=[W("get_x")]),
    ),
    (
        "AS-609",
        dict(
            src=code(tool_src("get_x", "x: Annotated[int, Field(le=5)]")), wire=[W("get_x", {"x": {"type": "integer"}})]
        ),
        dict(
            src=code(tool_src("get_x", "x: Annotated[int, Field(le=5)]")),
            wire=[W("get_x", {"x": {"type": "integer", "maximum": 5}})],
        ),
    ),
    # ---- Level-2 hygiene (no spec needed: risk_level=medium defaults)
    (
        "AS-219",
        dict(
            src=code(
                tool_src("get_x", body="try:\n        y = 1 / 0\n    except Exception:\n        pass\n    return 'ok'")
            )
        ),
        dict(
            src=code(
                tool_src(
                    "get_x",
                    body="try:\n        y = 1 / 0\n    except Exception as e:\n"
                    "        return f'failed: {e}'\n    return 'ok'",
                )
            )
        ),
    ),
    (
        "AS-220",
        dict(
            src=code(tool_src("update_system_prompt", "text: str", body="agent.instructions = text\n    return 'ok'"))
        ),
        dict(src=code(tool_src("get_x"))),
    ),
    (
        "AS-221",
        dict(src=code(tool_src("get_x"))),
        dict(
            src=code(
                tool_src("get_x"), extra="agent.run_sync('x', usage_limits=UsageLimits(total_tokens_limit=20000))\n"
            )
        ),
    ),
    (
        "AS-222",
        dict(src=code(tool_src("get_x"))),
        dict(src=code(tool_src("get_x"), extra="ex = AgentExecutor(agent=a, tools=[get_x], max_execution_time=60)\n")),
    ),
    (
        "AS-223",
        dict(src=code(tool_src("get_x"))),
        dict(src=code(tool_src("get_x"), extra="import os\nkey = os.environ['OPENAI_API_KEY']\n")),
    ),
    (
        "AS-224",
        dict(src=code(tool_src("get_x"))),
        dict(src=code(tool_src("get_x"), extra="agent = create_react_agent(llm, [get_x], response_format=Answer)\n")),
    ),
    (
        "AS-225",
        dict(src=code(tool_src("get_x"))),
        dict(src="import os\n" + code(tool_src("get_x"), extra="os.environ['LANGSMITH_TRACING'] = 'true'\n")),
    ),
    (
        "AS-226",
        dict(src=code(tool_src("get_x")), spec_text=spec(risk="high")),
        dict(
            src=code(
                tool_src("get_x"),
                extra="from langgraph.checkpoint.memory import MemorySaver\n"
                "graph = builder.compile(checkpointer=MemorySaver())\n",
            ),
            spec_text=spec(risk="high"),
        ),
    ),
    (
        "AS-227",
        dict(src=code(tool_src("submit_order", "order_id: str", body=WRITES)), spec_text=spec(risk="high")),
        dict(
            src=code(
                tool_src(
                    "submit_order",
                    "order_id: str, dry_run: bool = True",
                    body="if dry_run:\n        return 'preview'\n    " + WRITES,
                )
            ),
            spec_text=spec(risk="high"),
        ),
    ),
    (
        "AS-232",  # tool_effects: says read, the body writes
        dict(
            src=code(tool_src("submit_order", "order_id: str", body=WRITES)),
            spec_text=spec(extra="tool_effects: {submit_order: read}"),
        ),
        dict(
            src=code(tool_src("submit_order", "order_id: str", body=WRITES)),
            spec_text=spec(extra="tool_effects: {submit_order: write}"),
        ),
    ),
    ("AS-907", dict(src=code(tool_src("get_x"))), dict(src=code(tool_src("get_x")), spec_text=spec())),
    (
        "AS-901",
        dict(
            src="from smolagents import CodeAgent, tool\n"
            + tool_src("get_x")
            + "agent = CodeAgent(tools=[get_x], model=m)\n"
        ),
        dict(src=code(tool_src("get_x"))),
    ),
    ("AS-902", dict(src=code(tool_src("get_x")) + "\ndef broken(:\n"), dict(src=code(tool_src("get_x")))),
    ("AS-903", dict(src=code(tool_src("get_x"))), dict(src=code(tool_src("get_x")), wire=[W("get_x")])),
    ("AS-904", dict(src="def f():\n    return 1\n"), dict(src=code(tool_src("get_x")))),
    (
        "AS-906",
        dict(src=code(tool_src("get_x")), wire=[W("get_x"), {"name": "hosted:web_search", "parameters": {}}]),
        dict(src=code(tool_src("get_x")), wire=[W("get_x")]),
    ),
]


@pytest.mark.parametrize("rule_id,pos,neg", CASES, ids=[c[0] for c in CASES])
def test_rule_positive_and_negative(tmp_path: Path, rule_id: str, pos: dict[str, Any], neg: dict[str, Any]) -> None:
    (tmp_path / "p").mkdir()
    (tmp_path / "n").mkdir()
    src_p = pos.pop("src")
    src_n = neg.pop("src")
    if rule_id == "AS-906":  # hosted tools arrive via the hook's wire parser, emulate that here
        from agenticreq.models.observed import ToolSchema

        (tmp_path / "p" / "agent.py").write_text(src_p)
        call = chat_call(0, [W("get_x")])
        call.tools.append(ToolSchema(name="hosted:web_search", source="wire", kind="wire:hosted"))
        obs = build_observed(tmp_path / "p", captures=[call])
        got_p = {f.rule_id for f in run(mode="audit", observed=obs).findings}
    else:
        got_p = findings(tmp_path / "p", src_p, **pos)
    got_n = findings(tmp_path / "n", src_n, **neg)
    assert rule_id in got_p, f"positive: {rule_id} not in {sorted(got_p)}"
    assert rule_id not in got_n, f"negative: {rule_id} unexpectedly in {sorted(got_n)}"


COVERED_ELSEWHERE = {
    # AS-1xx: tests/test_phase2_spec.py (17 broken specs positive; good.yml is the negative for all)
    **{f"AS-1{n:02d}": "phase2" for n in range(1, 18)},
    "AS-100": "spec_status",
    "AS-118": "spec_status",
    "AS-201": "phase4",
    "AS-202": "phase4",
    "AS-204": "phase7",
    "AS-301": "phase6",
    "AS-302": "phase6",
    "AS-303": "phase6",
    "AS-304": "phase6",
    "AS-305": "phase6",
    "AS-306": "phase6/below",
    "AS-307": "phase6/below",
    "AS-308": "phase6/below",
    "AS-402": "phase7",
    "AS-905": "phase7",
    "AS-909": "test_agent_expansion.py",
    "AS-908": "test_discover_versions.py",
    "AS-233": "test_spec_names.py",
    "AS-234": "test_name_matching.py",
    "AS-910": "test_runtime_agents.py",
    "AS-911": "test_subfields.py",
    "AS-119": "test_output_polish.py",
    **{f"AS-{n}": "test_runtime_rules.py" for n in (228, 229, 230, 231)},
}


def test_every_rule_has_tests() -> None:
    _load_rule_modules()
    here = {c[0] for c in CASES}
    missing = [m.rule_id for m in REGISTRY.all() if m.rule_id not in here and m.rule_id not in COVERED_ELSEWHERE]
    assert not missing, missing


def test_multi_agent_hygiene_pos_neg(tmp_path: Path) -> None:
    from agenticreq.capture.discover import load_captures

    cap = load_captures(Path(__file__).parent / "fixtures" / "captures" / "oai_agents_po.json")
    obs = build_observed(tmp_path, captures=cap)
    base = spec(extra="agent_registry: [{id: x}]\n")
    got = {f.rule_id for f in run(mode="check", loaded=load_spec_text(base), observed=obs).findings}
    # captures only: the agents have no name in code, so they are a sign-off (AS-910), never "register a5119f"
    assert {"AS-306", "AS-307", "AS-910"} <= got and "AS-308" not in got
    single = build_observed(tmp_path)  # no delegation at all
    got_n = {f.rule_id for f in run(mode="check", loaded=load_spec_text(spec()), observed=single).findings}
    assert not {"AS-306", "AS-307", "AS-308"} & got_n


def test_detector_extension_point(tmp_path: Path) -> None:
    from agenticreq.traces import load_traces

    fix = Path(__file__).parent / "fixtures"
    spans = load_traces(fix / "traces" / "ap_agent.otlp.json")
    (tmp_path / "agent.py").write_text(code(tool_src("pay_invoice", "invoice_id: str, amount: float")))
    obs = build_observed(tmp_path, spans=spans)
    hit = spec(
        prohibited='[{action: "pay INV-9", reason: r, detector: {type: tool_arg_regex, '
        'params: {tool: pay_invoice, arg: invoice_id, pattern: "INV-9"}}}]'
    )
    miss = hit.replace("INV-9", "INV-0")
    r_hit = run(mode="check", loaded=load_spec_text(hit), observed=obs)
    f = next(f for f in r_hit.findings if f.rule_id == "AS-204")
    assert str(f.confidence) == "observed (1/3)" and "INV-9" in f.evidence
    r_miss = run(mode="check", loaded=load_spec_text(miss), observed=obs)
    assert not any(f.rule_id == "AS-204" for f in r_miss.findings)
    assert any(p.rule_id == "AS-204" and p.wording.startswith("no violation observed") for p in r_miss.passed)


def test_as604_suppresses_as603_on_same_field(tmp_path: Path) -> None:
    src = code(tool_src("get_x", "x: str = 'a', y: int = 1"))
    wire = [W("get_x", {"x": {"type": "string", "default": "a"}, "y": {"type": "integer"}}, ["x", "y"])]
    (tmp_path / "agent.py").write_text(src)
    obs = build_observed(tmp_path, captures=[chat_call(0, wire)])
    fs = run(mode="audit", observed=obs).findings
    by = {(f.rule_id, f.title.split("(")[-1].rstrip(")")) for f in fs if f.rule_id in ("AS-603", "AS-604")}
    assert by == {("AS-604", "x"), ("AS-603", "y")}  # x: one finding, not two
    f = next(f for f in fs if f.rule_id == "AS-604")
    assert f.severity == "blocker"
    assert f.evidence.endswith("wire: required=true, default='a' | code: optional, default='a'")
