"""Gaps found by a mutation run on a real Groq agent (agenticreq-full/mutation_report.txt), each now
closed: one positive (the gap) and one negative (the safe construct stays quiet) per fix."""

from __future__ import annotations

import textwrap
from datetime import date, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.ast import extract
from agenticreq.cli import app
from agenticreq.engine import run
from agenticreq.fix import plan
from agenticreq.models.finding import Finding
from agenticreq.observe import build_observed

H = "import requests, urllib.request\nfrom langchain_core.tools import tool\nBASE = 'https://api.example.com'\n"
FAKE_GROQ = "gsk_" + "0" * 52  # shape of a Groq key; built here so no key-like literal sits in this file


def _ids(tmp: Path, src: str) -> set[str]:
    (tmp / "agent.py").write_text(textwrap.dedent(src), encoding="utf-8")
    return {f.rule_id for f in run(mode="audit", observed=build_observed(tmp)).findings}


def _tool(name: str, params: str, body: str) -> str:
    return H + f"@tool\ndef {name}({params}) -> str:\n    '''Do.'''\n" + textwrap.indent(textwrap.dedent(body), "    ")


# ------------------------------------------------------------------ 1. AS-216: URL through a local variable
@pytest.mark.parametrize(
    "body,fires",
    [
        ("u = url\nreturn requests.get(u).text\n", True),
        (
            "req = urllib.request.Request(url, headers={'User-Agent': 'x'})\nreturn urllib.request.urlopen(req).read()\n",
            True,
        ),
        ("return requests.get(f'https://{url}/v1').text\n", True),  # the parameter is the host
        ("return requests.get(f'{BASE}/search?q={url}').text\n", False),  # fixed host, only the query varies
        ("return requests.get('https://api.example.com/' + url).text\n", False),  # fixed host, path varies
        ("q = f'{BASE}/latest?to={url}'\nreturn urllib.request.urlopen(q).read()\n", False),
    ],
)
def test_as216_follows_the_host(tmp_path: Path, body: str, fires: bool) -> None:
    assert ("AS-216" in _ids(tmp_path, _tool("fetch_url", "url: str", body))) is fires


# ------------------------------------------------------------------ 2. AS-213: body evidence, not only the name
@pytest.mark.parametrize(
    "body,fires",
    [
        ("cur = db.cursor()\ncur.execute('SELECT * FROM stories')\nreturn str(cur.fetchall())\n", True),
        ("return str(session.query(Story).filter(Story.ok).all())\n", True),
        ("cur = db.cursor()\ncur.execute('SELECT * FROM stories LIMIT 10')\nreturn str(cur.fetchall())\n", False),
        (
            "cur = db.cursor()\ncur.execute('SELECT * FROM stories WHERE id = ?', (x,))\nreturn str(cur.fetchone())\n",
            False,
        ),
        ("cur = db.cursor()\ncur.execute('SELECT COUNT(*) FROM stories')\nreturn str(cur.fetchall())\n", False),
    ],
)
def test_as213_reads_the_body(tmp_path: Path, body: str, fires: bool) -> None:
    # `top_stories` is not a search_/list_/find_ name: only the body can tell
    assert ("AS-213" in _ids(tmp_path, _tool("top_stories", "x: str", body))) is fires


# ------------------------------------------------------------------ 3 + 4. AS-221 budget must be passed; AS-225 needs a tool-level hook
BUDGET = """
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

class TokenBudget(BaseCallbackHandler):
    def __init__(self, max_total_tokens): self.left = max_total_tokens
    def on_llm_end(self, response, **kw): self.left -= 1

class ToolAudit(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kw): log.info(input_str)

@tool
def lookup(q: str) -> str:
    '''Look.'''
    return q

graph = create_react_agent("gpt-4o-2024-08-06", tools=[lookup])
budget = TokenBudget(max_total_tokens=20000)
"""


@pytest.mark.parametrize(
    "tail,cost,trace",
    [
        ("graph.invoke({}, config={'recursion_limit': 5})\n", True, True),  # built, never passed
        (
            "graph.invoke({}, config={'recursion_limit': 5, 'callbacks': [budget]})\n",
            False,
            True,
        ),  # passed; counts tokens only
        ("graph.invoke({}, config={'recursion_limit': 5, 'callbacks': [budget, ToolAudit()]})\n", False, False),
        (
            "from langchain_core.tracers import LangChainTracer\n"
            "graph.invoke({}, config={'recursion_limit': 5, 'callbacks': [budget, LangChainTracer()]})\n",
            False,
            False,
        ),
    ],
)
def test_budget_must_be_used_and_token_counter_is_not_an_audit_hook(
    tmp_path: Path, tail: str, cost: bool, trace: bool
) -> None:
    ids = _ids(tmp_path, BUDGET + tail)
    assert ("AS-221" in ids) is cost
    assert ("AS-225" in ids) is trace


# ------------------------------------------------------------------ 5. AS-214: Groq key in any position
@pytest.mark.parametrize(
    "line,fires",
    [
        (f"K = {FAKE_GROQ!r}\n", True),  # an ordinary variable name
        (f"client = Groq(api_key={FAKE_GROQ!r})\n", True),  # a kwarg
        (f"headers = {{'Authorization': 'Bearer {FAKE_GROQ}'}}\n", True),  # inside a header value
        ("K = 'gsk_short'\n", False),  # too short to be a key
        ("headers = {'Authorization': 'Bearer ' + os.environ['GROQ_API_KEY']}\n", False),
    ],
)
def test_as214_groq_keys(tmp_path: Path, line: str, fires: bool) -> None:
    assert ("AS-214" in _ids(tmp_path, "import os\n" + line)) is fires


# ------------------------------------------------------------------ 6. audit --spec applies waivers only
AGENT_UNPINNED = (
    "from langchain_core.tools import tool\nfrom langchain_openai import ChatOpenAI\n"
    "@tool\ndef lookup(q: str) -> str:\n    '''Look.'''\n    return q\n"
    "llm = ChatOpenAI(model='gpt-4o').bind_tools([lookup])\n"
)


@pytest.mark.parametrize(
    "expires,waived", [(date.today() + timedelta(days=30), True), (date.today() - timedelta(days=1), False)]
)
def test_audit_spec_applies_unexpired_waivers(tmp_path: Path, expires: date, waived: bool) -> None:
    import json

    (tmp_path / "agent.py").write_text(AGENT_UNPINNED, encoding="utf-8")
    spec = tmp_path / "spec.yml"
    spec.write_text(
        f"agent_id: a\nwaivers:\n  - rule_id: AS-210\n    reason: provider has no snapshots\n"
        f"    approved_by: lead\n    expires: {expires.isoformat()}\n",
        encoding="utf-8",
    )

    def rules(*extra: str) -> set[str]:
        r = CliRunner().invoke(app, ["audit", str(tmp_path), "-f", "json", *extra])
        d = json.loads(r.stdout)
        return {f["rule_id"] for f in d["findings"]}

    assert "AS-210" in rules()  # no --spec: unchanged
    assert ("AS-210" in rules("--spec", str(spec))) is not waived
    assert "AS-907" in rules("--spec", str(spec))  # Level 1 is still not checked by audit


# ------------------------------------------------------------------ 7. AS-210 knows providers without snapshots
@pytest.mark.parametrize(
    "model_line,groq",
    [
        ("llm = ChatGroq(model='llama-3.3-70b-versatile')", True),
        ("llm = ChatOpenAI(model='openai/gpt-oss-120b', base_url='https://api.groq.com/openai/v1')", True),
        (
            "GROQ_BASE_URL = os.environ['GROQ_BASE_URL']\n"
            "llm = ChatOpenAI(model='openai/gpt-oss-120b', base_url=GROQ_BASE_URL)",
            True,
        ),
        ("llm = LiteLLMModel(model_id='groq/llama-3.3-70b-versatile')", True),
        ("llm = ChatOpenAI(model='gpt-4o')", False),  # OpenAI: pin a snapshot
    ],
)
def test_as210_provider_without_snapshots(tmp_path: Path, model_line: str, groq: bool) -> None:
    src = (
        "import os\nfrom langchain_core.tools import tool\n@tool\ndef lookup(q: str) -> str:\n"
        "    '''Look.'''\n    return q\n" + model_line + "\n"
    )
    (tmp_path / "agent.py").write_text(src, encoding="utf-8")
    obs = build_observed(tmp_path)
    f = next(f for f in run(mode="audit", observed=obs).findings if f.rule_id == "AS-210")
    if groq:
        assert f.severity == "info" and "offers no model snapshots" in f.title
        assert f.bucket.value == "UNKNOWNS"  # never a PASS, never the impossible "pin it" advice
        assert not [p for p in plan(extract(tmp_path), tmp_path, 25) if p.rule_id == "AS-210"]
    else:
        assert f.severity == "warning" and "not pinned" in f.title


# ================================================================== round 2: framework mutation run (10 gaps)
def _audit(tmp: Path, src: str, spec: str | None = None) -> list[Finding]:
    from agenticreq.loader import load_spec_text

    (tmp / "agent.py").write_text(textwrap.dedent(src), encoding="utf-8")
    loaded = load_spec_text(textwrap.dedent(spec)) if spec else None
    return run(mode="check" if loaded else "audit", loaded=loaded, observed=build_observed(tmp)).findings


# ---- 1. a tool's own needs_approval / requires_approval gates THAT tool only
@pytest.mark.parametrize(
    "src",
    [
        """
    import requests
    from agents import Agent, function_tool
    @function_tool(needs_approval=True)
    def issue_refund(order_id: str) -> str:
        '''Refund.'''
        return order_id
    @function_tool
    def send_email(to: str) -> str:
        '''Email.'''
        requests.post('https://mail.example/send', json={'to': to})  # a real write: the body decides
        return to
    agent = Agent(name="a", instructions="i", tools=[issue_refund, send_email])
    """,
        """
    import requests
    from pydantic_ai import Agent
    agent = Agent("openai:gpt-4o-2024-08-06")
    @agent.tool_plain(requires_approval=True)
    def issue_refund(order_id: str) -> str:
        '''Refund.'''
        return order_id
    @agent.tool_plain
    def send_email(to: str) -> str:
        '''Email.'''
        requests.post('https://mail.example/send', json={'to': to})  # a real write: the body decides
        return to
    """,
    ],
)
def test_tool_scoped_approval_does_not_cover_other_tools(tmp_path: Path, src: str) -> None:
    hits = [f for f in _audit(tmp_path, src) if f.rule_id == "AS-212"]
    assert [f.title for f in hits] == ["Write tool send_email runs without approval"]


# ---- 2. CrewAI Task(human_input=True) reviews the answer after the tools ran: not an approval gate
CREW = """
import requests
from crewai import Agent, Task
from crewai.tools import tool
@tool("pay_vendor")
def pay_vendor(vendor: str) -> str:
    '''Pay.'''
    requests.post("https://pay.example/vendors", json={"vendor": vendor})  # a real payment: a write by its body
    return vendor
a = Agent(role="r", goal="g", backstory="b", tools=[pay_vendor], max_iter=5)
t = Task(description="d", expected_output="e", agent=a, human_input=True)
"""
SPEC_APPROVAL = """
agent_id: a
risk_level: medium
goal: {statement: g, done_criteria: [d]}
out_of_scope: []
prohibited: []
approval_required: [{tool: pay_vendor, approver: lead, timeout: 1h, on_timeout: safe_stop}]
limits: []
escalation: []
untrusted_inputs: []
owner: o
technical_owner: t
"""
NO_APPROVAL = SPEC_APPROVAL.replace(
    "approval_required: [{tool: pay_vendor, approver: lead, timeout: 1h, on_timeout: safe_stop}]",
    "approval_required: []",
)


def test_crewai_task_human_input_is_not_an_approval_gate(tmp_path: Path) -> None:
    f = next(f for f in _audit(tmp_path, CREW, SPEC_APPROVAL) if f.rule_id == "AS-205")
    assert "output_review" in f.evidence


# ---- 3. agent_registry: an agent's own prohibited / listed tools
REG_AGENT = """
from crewai import Agent
from crewai.tools import tool
@tool("query_db")
def query_db(q: str) -> str:
    '''Query.'''
    return q
@tool("write_report")
def write_report(text: str) -> str:
    '''Write.'''
    return text
researcher = Agent(role="researcher", goal="g", backstory="b", tools=[query_db], max_iter=5)
writer = Agent(role="writer", goal="g", backstory="b", tools=[write_report, query_db], max_iter=5)
"""


@pytest.mark.parametrize(
    "entry,expect",
    [
        ("{id: writer, tools: [write_report, query_db], prohibited: [query_db]}", "prohibited"),
        ("{id: writer, tools: [write_report]}", "does not list"),
        ("{id: writer, tools: [write_report, query_db]}", None),
    ],
)
def test_registry_entry_is_enforced_on_the_agent_itself(tmp_path: Path, entry: str, expect: str | None) -> None:
    spec = NO_APPROVAL + f"topology: pipeline\nagent_registry: [{{id: researcher, tools: [query_db]}}, {entry}]\n"
    titles = [
        f.title
        for f in _audit(tmp_path, REG_AGENT, spec)
        if f.rule_id in ("AS-203", "AS-206") and f.title.startswith("Agent writer")
    ]
    if expect is None:
        assert not titles
    else:
        assert titles and all(expect in t for t in titles)


# ---- 4 + 5. PydanticAI GroqModel, agents-SDK client objects
@pytest.mark.parametrize(
    "src",
    [
        """
    from pydantic_ai import Agent
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider
    model = GroqModel("llama-3.3-70b-versatile", provider=GroqProvider(api_key=key))
    agent = Agent(model)
    @agent.tool_plain
    def lookup(q: str) -> str:
        '''Look.'''
        return q
    """,
        """
    from agents import Agent, OpenAIChatCompletionsModel, function_tool
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
    @function_tool
    def lookup(q: str) -> str:
        '''Look.'''
        return q
    agent = Agent(name="a", instructions="i", tools=[lookup],
                  model=OpenAIChatCompletionsModel(model="openai/gpt-oss-120b", openai_client=client))
    """,
        """
    from agents import Agent, function_tool, set_default_openai_client
    from openai import AsyncOpenAI
    set_default_openai_client(AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=key))
    @function_tool
    def lookup(q: str) -> str:
        '''Look.'''
        return q
    agent = Agent(name="a", instructions="i", tools=[lookup], model="openai/gpt-oss-120b")
    """,
    ],
)
def test_groq_recognised_through_framework_objects(tmp_path: Path, src: str) -> None:
    f = next(f for f in _audit(tmp_path, src) if f.rule_id == "AS-210")
    assert "offers no model snapshots" in f.title


def test_pydantic_ai_decorator_tools_are_bound_to_their_agent(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        textwrap.dedent("""
        from pydantic_ai import Agent
        agent = Agent("openai:gpt-4o-2024-08-06")
        @agent.tool_plain
        def lookup(q: str) -> str:
            '''Look.'''
            return q
    """),
        encoding="utf-8",
    )
    obs = build_observed(tmp_path)
    assert "lookup" in obs.bound_tools
    assert "lookup" in next(a for a in obs.declared_agents if a.var == "agent").tools


# ---- 6. fail-open inside a class tool's _run
@pytest.mark.parametrize("handler,fires", [("pass", True), ("raise", False)])
def test_fail_open_in_class_tool(tmp_path: Path, handler: str, fires: bool) -> None:
    src = f"""
    from langchain_core.tools import BaseTool
    class LookupTool(BaseTool):
        name: str = "lookup"
        description: str = "Look."
        def _run(self, q: str) -> str:
            try:
                return db.get(q)
            except Exception:
                {handler}
    """
    assert ("AS-219" in {f.rule_id for f in _audit(tmp_path, src)}) is fires


# ---- 7. validating one model call's JSON against a schema is structured output, but not shown to be the
# final answer on every path: UNKNOWN, never a PASS (and never the "no structured output" warning)
@pytest.mark.parametrize(
    "line,structured",
    [
        ("report = Report.model_validate_json(raw)", True),
        ("report = TypeAdapter(Report).validate_json(raw)", True),
        ("report = json.loads(raw)", False),
    ],
)
def test_schema_validation_counts_as_structured_output(tmp_path: Path, line: str, structured: bool) -> None:
    src = (
        "import json\nfrom langchain_core.tools import tool\nfrom langchain_openai import ChatOpenAI\n"
        "@tool\ndef lookup(q: str) -> str:\n    '''Look.'''\n    return q\n"
        "llm = ChatOpenAI(model='gpt-4o-2024-08-06').bind_tools([lookup])\nraw = llm.invoke('x').content\n"
        + line
        + "\n"
    )
    f = next(f for f in _audit(tmp_path, src) if f.rule_id == "AS-224")
    assert (f.confidence.level.value == "unknown") is structured


# ---- 9. UsageLimits caps spend only with a token limit
@pytest.mark.parametrize(
    "limits,cost_cap",
    [("UsageLimits(request_limit=6)", False), ("UsageLimits(request_limit=6, total_tokens_limit=20000)", True)],
)
def test_usage_limits_without_tokens_is_not_a_cost_cap(tmp_path: Path, limits: str, cost_cap: bool) -> None:
    src = f"""
    from pydantic_ai import Agent
    from pydantic_ai.usage import UsageLimits
    agent = Agent("openai:gpt-4o-2024-08-06")
    @agent.tool_plain
    def lookup(q: str) -> str:
        '''Look.'''
        return q
    agent.run_sync("x", usage_limits={limits})
    """
    assert ("AS-221" in {f.rule_id for f in _audit(tmp_path, src)}) is not cost_cap


# ---- 10. a medium spec is never quieter than no spec for high-impact writes
def test_medium_spec_keeps_high_impact_write_gate(tmp_path: Path) -> None:
    src = CREW.replace(", human_input=True", "")
    assert "AS-212" in {f.rule_id for f in _audit(tmp_path, src)}
    assert "AS-212" in {f.rule_id for f in _audit(tmp_path, src, NO_APPROVAL)}


# ---- spec: deny on approval timeout is safe; a declared delegation cycle contradicts no_cycles
@pytest.mark.parametrize(
    "risk,action,flagged",
    [("high", "deny", False), ("critical", "deny", False), ("high", "escalate", True), ("medium", "proceed", True)],
)
def test_on_timeout_deny_is_safe(risk: str, action: str, flagged: bool) -> None:
    from agenticreq.loader import load_spec_text

    spec = (
        SPEC_APPROVAL.replace("risk_level: medium", f"risk_level: {risk}").replace(
            "on_timeout: safe_stop", f"on_timeout: {action}"
        )
        + f"on_approval_timeout: {action}\n"
    )
    found = run(mode="validate", loaded=load_spec_text(spec)).findings
    assert any(f.rule_id == "AS-110" for f in found) is flagged
    assert any(f.rule_id == "AS-111" and "on_approval_timeout" in f.detail for f in found) is flagged


@pytest.mark.parametrize(
    "gate_line,where,covers_other",
    [
        ("request_approval(f'pay {vendor}')", "tool", False),  # inside pay_vendor: gates pay_vendor only
        ("interrupt('approve?')", "tool", False),
        ("interrupt('approve?')", "node", True),  # a graph node: gates whatever runs there
    ],
)
def test_gate_inside_a_tool_scopes_to_that_tool(tmp_path: Path, gate_line: str, where: str, covers_other: bool) -> None:
    body = f"    {gate_line}\n" if where == "tool" else ""
    node = f"\ndef review(state):\n    {gate_line}\n    return state\n" if where == "node" else ""
    src = (
        "from langchain_core.tools import tool\nfrom langgraph.types import interrupt\n"
        "from langchain_openai import ChatOpenAI\n"
        f"@tool\ndef pay_vendor(vendor: str) -> str:\n    '''Pay.'''\n{body}    return vendor\n"
        "import requests\n@tool\ndef send_email(to: str) -> str:\n    '''Email.'''\n"
        "    requests.post('https://mail.example/send', json={'to': to})\n    return to\n"
        + node
        + "llm = ChatOpenAI(model='gpt-4o-2024-08-06').bind_tools([pay_vendor, send_email])\n"
    )
    titles = {f.title for f in _audit(tmp_path, src) if f.rule_id == "AS-212"}
    assert ("Write tool send_email runs without approval" in titles) is not covers_other
    assert "Write tool pay_vendor runs without approval" not in titles


def test_declared_delegation_cycle(tmp_path: Path) -> None:
    spec = SPEC_APPROVAL + (
        "topology: peer\nagent_registry: [{id: a}, {id: b}]\n"
        "declared_delegation: [{from: a, to: b}, {from: b, to: a}]\n"
    )
    titles = [f.title for f in _audit(tmp_path, CREW, spec) if f.rule_id == "AS-303"]
    assert any("declared_delegation contains a cycle" in t for t in titles)
