"""Validation run 3 (10 public repos) gaps, the GitHub Action / SARIF paths, and the smarter `init`.
Each fix: a positive (the construct found in a real repo) and a negative (what must stay quiet)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from agenticreq.cli import app
from agenticreq.engine import run
from agenticreq.models.finding import Finding
from agenticreq.observe import build_observed

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


def audit(tmp: Path, src: str, extra_files: dict[str, str] | None = None) -> tuple[list[Finding], set[str]]:
    (tmp / "agent.py").write_text(textwrap.dedent(src), encoding="utf-8")
    for name, text in (extra_files or {}).items():
        (tmp / name).write_text(text, encoding="utf-8")
    r = run(mode="audit", observed=build_observed(tmp))
    return r.findings, {p.rule_id for p in r.passed}


def ids(fs: list[Finding]) -> set[str]:
    return {f.rule_id for f in fs}


LG = """
import os
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def lookup(q: str) -> str:
    '''Look.'''
    return q

graph = create_react_agent("gpt-4o-2024-08-06", tools=[lookup])
"""


# ------------------------------------------------------------------ AS-225: tracing that is not a callbacks= kwarg
@pytest.mark.parametrize(
    "extra,files,traced",
    [
        (
            "from monocle_apptrace import setup_monocle_telemetry\nsetup_monocle_telemetry(workflow_name='x')\n",
            {},
            True,
        ),
        ("os.environ['LANGCHAIN_TRACING_V2'] = 'true'\n", {}, True),
        ("", {".env.example": "LANGSMITH_TRACING=true\nLANGSMITH_API_KEY=\n"}, True),
        ("", {".env.example": "LANGSMITH_TRACING=false\n"}, False),
        ("os.environ['LANGCHAIN_TRACING_V2'] = 'false'\n", {}, False),
    ],
)
def test_tracing_switched_on_outside_callbacks(tmp_path: Path, extra: str, files: dict[str, str], traced: bool) -> None:
    fs, passed = audit(tmp_path, LG + extra, files)
    assert ("AS-225" in ids(fs)) is not traced


AGENTS = """
from agents import Agent, Runner, function_tool{imp}
@function_tool
def lookup(q: str) -> str:
    '''Look.'''
    return q
agent = Agent(name="a", instructions="i", tools=[lookup], model="gpt-4o-2024-08-06")
{body}
"""


@pytest.mark.parametrize(
    "imp,body,traced",
    [
        ("", "Runner.run_sync(agent, 'x')", True),  # SDK traces by default
        (", trace", "with trace('wf'):\n    Runner.run_sync(agent, 'x')", True),
        (", set_tracing_disabled", "set_tracing_disabled(True)\nRunner.run_sync(agent, 'x')", False),
        (
            ", set_tracing_disabled, trace",
            "set_tracing_disabled(True)\nwith trace('wf'):\n    Runner.run_sync(agent, 'x')",
            False,
        ),  # trace() goes nowhere once disabled
    ],
)
def test_openai_agents_tracing_default(tmp_path: Path, imp: str, body: str, traced: bool) -> None:
    fs, _ = audit(tmp_path, AGENTS.format(imp=imp, body=body))
    assert ("AS-225" in ids(fs)) is not traced


# ------------------------------------------------------------------ AS-208: the project's own loop counter
@pytest.mark.parametrize(
    "route,bounded",
    [
        (
            "def route(state, config):\n    if state.loop_step < config['configurable']['max_loops']:\n"
            "        return 'agent'\n    return END\n",
            True,
        ),
        ("def route(state):\n    if state.messages[-1].tool_calls:\n        return 'tools'\n    return END\n", False),
    ],
)
def test_own_loop_counter_is_a_step_bound(tmp_path: Path, route: str, bounded: bool) -> None:
    # a counter compared with a limit is seen, but not verified to stop every path: UNKNOWN, never a PASS
    fs, passed = audit(tmp_path, LG.replace("graph = create_react_agent", route + "\ngraph = create_react_agent"))
    f = next(f for f in fs if f.rule_id == "AS-208")
    assert (f.confidence.level.value == "unknown") is bounded
    assert "AS-208" not in passed


# ------------------------------------------------------------------ AS-223: configuration injected at runtime
@pytest.mark.parametrize(
    "extra,files,separated",
    [
        ("def node(state, config):\n    model = config['configurable']['model']\n", {}, True),
        ("", {"langgraph.json": json.dumps({"graphs": {"a": "./agent.py:graph"}, "env": ".env"})}, True),
        ("", {}, False),  # nothing comes from the environment or the deployment
    ],
)
def test_runtime_configuration_counts_as_environment_separation(
    tmp_path: Path, extra: str, files: dict[str, str], separated: bool
) -> None:
    fs, _ = audit(tmp_path, LG.replace("import os\n", "") + extra, files)
    assert ("AS-223" in ids(fs)) is not separated


# ------------------------------------------------------------------ AS-222: only a timeout on the RUN bounds it
@pytest.mark.parametrize(
    "extra,bounded",
    [
        ("import requests\nrequests.get('https://x.org', timeout=5)\n", False),  # one HTTP call
        ("retriever.invoke('q', timeout=30)\n", False),  # a sub-call
        ("import asyncio\nasync def main():\n    await asyncio.wait_for(graph.ainvoke({}), 60)\n", True),
        (
            "import asyncio\nasync def main():\n    async with asyncio.timeout(60):\n        await graph.ainvoke({})\n",
            True,
        ),
        ("graph.invoke({}, timeout=60)\n", True),
    ],
)
def test_only_a_run_level_timeout_counts(tmp_path: Path, extra: str, bounded: bool) -> None:
    fs, _ = audit(tmp_path, LG + extra)
    assert ("AS-222" in ids(fs)) is not bounded


# ------------------------------------------------------------------ AS-219: fail-open on the agent's path
@pytest.mark.parametrize(
    "extra,fires",
    [
        # a LangGraph node that ends the run on any error (open_deep_research shape)
        (
            "from langgraph.graph import StateGraph, END\n"
            "def supervise(state):\n    try:\n        return work(state)\n    except Exception as e:\n"
            "        if over_limit(e) or True:\n            return Command(goto=END)\n"
            "b = StateGraph(dict)\nb.add_node('supervise', supervise)\n",
            True,
        ),
        # an orchestrator method that runs the agent and swallows the error (financial_research_agent shape)
        (
            "class Manager:\n    async def verify(self, text):\n        try:\n            return await Runner.run(graph, text)\n"
            "        except Exception:\n            return None\n",
            True,
        ),
        # a helper the tool calls, which prints and returns '' (stock_analysis / gpt-researcher shape)
        (
            "def fetch(q):\n    try:\n        return remote(q)\n    except Exception as e:\n        print(f'Error: {e}')\n"
            "        return ''\n",
            True,
        ),
        # retry loop: counts attempts and tries again
        (
            "def supervise(state):\n    attempts = 0\n    while attempts < 3:\n        try:\n            return work(state)\n"
            "        except Exception:\n            attempts += 1\n"
            "from langgraph.graph import StateGraph\nb = StateGraph(dict)\nb.add_node('supervise', supervise)\n",
            False,
        ),
    ],
)
def test_fail_open_beyond_tool_bodies(tmp_path: Path, extra: str, fires: bool) -> None:
    src = LG.replace("    return q\n", "    return fetch(q)\n" if "def fetch" in extra else "    return q\n") + extra
    fs, _ = audit(tmp_path, src)
    assert ("AS-219" in ids(fs)) is fires


# ------------------------------------------------------------------ AS-216: URL through a helper method
@pytest.mark.parametrize(
    "guard,fires",
    [("", True), ("    if not url.startswith('https://docs.x.com/'):\n        raise ValueError(url)\n", False)],
)
def test_url_through_helper(tmp_path: Path, guard: str, fires: bool) -> None:
    src = f"""
    import requests
    from smolagents import Tool
    class Browser:
        def visit_page(self, url):
    {textwrap.indent(guard, "    ")}        return requests.get(url).text
    class VisitTool(Tool):
        name = "visit_page"
        description = "Visit."
        inputs = {{"url": {{"type": "string", "description": "u"}}}}
        output_type = "string"
        def forward(self, url):
            return self.browser.visit_page(url)
    """
    fs, _ = audit(tmp_path, src)
    assert ("AS-216" in ids(fs)) is fires


# ------------------------------------------------------------------ AS-210: model id from a default / CLI / **params
@pytest.mark.parametrize(
    "src",
    [
        "def main(model_id='o1'):\n    model = LiteLLMModel(model_id=model_id)\n",
        "import argparse\np = argparse.ArgumentParser()\np.add_argument('--model-id', default='o1')\nargs = p.parse_args()\n"
        "model = LiteLLMModel(model_id=args.model_id)\n",
        "def main(model_id='o1'):\n    params = {'model_id': model_id, 'max_tokens': 10}\n    model = LiteLLMModel(**params)\n",
    ],
)
def test_unpinned_model_from_defaults(tmp_path: Path, src: str) -> None:
    head = (
        "from smolagents import LiteLLMModel, ToolCallingAgent, tool\n@tool\ndef lookup(q: str) -> str:\n"
        "    '''Look.\n\n    Args:\n        q: query.\n    '''\n    return q\n"
    )
    fs, _ = audit(tmp_path, head + src)
    assert any(f.rule_id == "AS-210" and "'o1'" in f.title for f in fs)


# ------------------------------------------------------------------ AS-215: agents that run model-written code
@pytest.mark.parametrize(
    "ctor,fires",
    [
        ("CodeAgent(tools=[], model=m, additional_authorized_imports=['*'])", True),
        ("CodeAgent(tools=[], model=m, additional_authorized_imports=['*'], executor_type='docker')", False),
        ("CodeAgent(tools=[], model=m)", False),  # restricted local interpreter
        ("UserProxyAgent('u', code_execution_config={'use_docker': False})", True),
        ("UserProxyAgent('u', code_execution_config=False)", False),
    ],
)
def test_code_executing_agents(tmp_path: Path, ctor: str, fires: bool) -> None:
    fs, _ = audit(tmp_path, f"from smolagents import CodeAgent\nfrom autogen import UserProxyAgent\nagent = {ctor}\n")
    assert ("AS-215" in ids(fs)) is fires


# ------------------------------------------------------------------ smarter init: suggestions, never values
def test_init_suggests_from_code_but_fills_nothing(tmp_path: Path) -> None:
    from agenticreq.params import level
    from agenticreq.scaffold import render_init

    (tmp_path / "agent.py").write_text(
        textwrap.dedent("""
        import subprocess
        import requests
        from langchain_core.tools import tool
        @tool
        def read_supplier_email(inbox: str) -> str:
            '''Read email.'''
            return inbox
        @tool
        def submit_order(sku: str, amount: float) -> str:
            '''Order.'''
            requests.post("https://shop.example/orders", json={"sku": sku, "amount": amount})
            return sku
        @tool
        def pay_invoice(invoice_id: str, card_number: str) -> str:
            '''Pay.'''
            requests.post("https://pay.example/charge", json={"id": invoice_id, "card": card_number})
            return invoice_id
        @tool
        def run_shell(cmd: str) -> str:
            '''Shell.'''
            return subprocess.run(cmd, shell=True).stdout
    """),
        encoding="utf-8",
    )
    text = render_init(build_observed(tmp_path))
    data = yaml.safe_load(text)
    for p in level(1):
        assert data[p.name] == "TODO", p.name  # suggestions are comments: check still fails until a human decides
    hints = "\n".join(line for line in text.splitlines() if "read from the code" in line)
    assert "high" in hints
    assert "{tool: run_shell" in hints and "{tool: pay_invoice" in hints  # prohibited candidates
    assert "{tool: submit_order, approver: TODO" in hints  # approval candidate
    assert "{tool: submit_order, arg: amount, limit: {max: TODO}}" in hints
    assert "{source: read_supplier_email}" in hints
    assert "never_store: [card_number]" in hints


def test_init_lists_registry_and_delegation_in_derived_for_multi_agent(tmp_path: Path) -> None:
    """Names are the user's requirement: init writes TODO and lists what the code has in _derived only."""
    from agenticreq.scaffold import render_init

    (tmp_path / "agent.py").write_text(
        textwrap.dedent("""
        from agents import Agent
        billing = Agent(name="billing", instructions="b")
        triage = Agent(name="triage", instructions="t", handoffs=[billing])
    """),
        encoding="utf-8",
    )
    text = render_init(build_observed(tmp_path))
    assert "agent_registry: TODO" in text and '"billing"' in text and "{id: triage" not in text
    assert 'delegation_in_code: ["triage -> billing"]' in text and "{from: triage, to: billing}" not in text


# ------------------------------------------------------------------ GitHub Action + SARIF + --fail-on
def test_sarif_uris_are_repo_relative_for_a_subfolder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = tmp_path / "agents" / "po"
    agent.mkdir(parents=True)
    (agent / "agent.py").write_text(LG.replace('"gpt-4o-2024-08-06"', '"gpt-4o"'), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["audit", "agents/po", "-f", "sarif", "-o", "out.sarif"])
    sarif = json.loads((tmp_path / "out.sarif").read_text(encoding="utf-8"))
    uris = {
        loc["physicalLocation"]["artifactLocation"]["uri"]
        for r in sarif["runs"][0]["results"]
        for loc in r.get("locations", [])
    }
    assert uris and all(u.startswith("agents/po/") and "\\" not in u for u in uris), uris


@pytest.mark.parametrize("fail_on,code", [("gaps", 1), ("all", 1), ("never", 0)])
def test_fail_on(tmp_path: Path, fail_on: str, code: int) -> None:
    (tmp_path / "agent.py").write_text(LG, encoding="utf-8")  # warnings only (missing), no blockers
    r = runner.invoke(app, ["audit", str(tmp_path), "-f", "json", "--fail-on", fail_on])
    assert r.exit_code == code


def test_action_definition_is_safe_and_complete() -> None:
    action = yaml.safe_load((ROOT / "action.yml").read_text(encoding="utf-8"))
    assert set(action["inputs"]) >= {"path", "spec", "captures", "traces", "fail-on", "sarif-file", "upload-sarif"}
    steps = action["runs"]["steps"]
    assert any("upload-sarif" in str(s.get("uses", "")) for s in steps)
    for s in steps:  # inputs reach shell scripts only through env:, never interpolated into run:
        assert "${{ inputs." not in s.get("run", ""), s.get("name")


def test_agents_sdk_default_tracing_needs_an_openai_model(tmp_path: Path) -> None:
    groq = AGENTS.format(imp=", OpenAIChatCompletionsModel", body="").replace(
        'model="gpt-4o-2024-08-06"',
        'model=OpenAIChatCompletionsModel(model="openai/gpt-oss-120b", '
        'openai_client=AsyncOpenAI(base_url="https://api.groq.com/openai/v1"))',
    )
    fs, _ = audit(tmp_path, "from openai import AsyncOpenAI\n" + groq)
    assert "AS-225" in ids(fs)  # default traces go to OpenAI's platform: nothing is recorded for a Groq model
    fs, _ = audit(tmp_path, AGENTS.format(imp="", body="import os\nos.environ['OPENAI_AGENTS_DISABLE_TRACING'] = '1'"))
    assert "AS-225" in ids(fs)
