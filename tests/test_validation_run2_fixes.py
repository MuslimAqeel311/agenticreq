"""Regression tests for the false positives found by validation run 2 (VALIDATION.md)."""

from pathlib import Path
from textwrap import dedent

from agenticreq.ast.classify import is_pinned, is_unbounded, pin_judge
from agenticreq.engine import run
from agenticreq.observe import build_observed


def _obs(tmp: Path, src: str):
    (tmp / "agent.py").write_text(dedent(src))
    return build_observed(tmp)


def test_search_capped_in_body_is_bounded(tmp_path: Path) -> None:
    """react-agent: search(query) wraps TavilySearch(max_results=...) — the body bounds the output."""
    obs = _obs(
        tmp_path,
        """
        from langchain_core.tools import tool
        from langchain_tavily import TavilySearch
        @tool
        def search(query: str) -> dict:
            \"\"\"Search.\"\"\"
            return TavilySearch(max_results=5).invoke({"query": query})
        @tool
        def find_rows(q: str) -> list:
            \"\"\"Find.\"\"\"
            return db(q)[:20]
        @tool
        def list_everything(q: str) -> list:
            \"\"\"List.\"\"\"
            return db(q)
    """,
    )
    assert not is_unbounded(obs.declared_tools["search"])
    assert not is_unbounded(obs.declared_tools["find_rows"])
    assert is_unbounded(obs.declared_tools["list_everything"])


def test_param_named_context_with_plain_type_is_kept(tmp_path: Path) -> None:
    """memory-agent: `context: str` is a model-visible argument; `config: RunnableConfig` is injected."""
    obs = _obs(
        tmp_path,
        """
        from typing import Annotated
        from langchain_core.tools import tool, InjectedToolArg
        from langchain_core.runnables import RunnableConfig
        @tool
        def upsert_memory(content: str, context: str, *, config: RunnableConfig,
                          user_id: Annotated[str, InjectedToolArg]) -> str:
            \"\"\"Upsert.\"\"\"
            return ""
    """,
    )
    assert [p.name for p in obs.declared_tools["upsert_memory"].params] == ["content", "context"]


def test_agent_as_tool_has_input_param(tmp_path: Path) -> None:
    obs = _obs(
        tmp_path,
        """
        from agents import Agent
        a = Agent(name="A", instructions="x")
        t = a.as_tool(tool_name="analyse", tool_description="d")
    """,
    )
    assert [(p.name, p.type, p.required) for p in obs.declared_tools["analyse"].params] == [("input", "string", True)]


def test_openai_agents_run_bound_and_tracing_cover_the_system(tmp_path: Path) -> None:
    obs = _obs(
        tmp_path,
        """
        from agents import Agent
        b = Agent(name="B", instructions="b")
        a = Agent(name="A", instructions="a", handoffs=[b])
    """,
    )
    r = run(mode="audit", observed=obs)
    assert "AS-306" not in {f.rule_id for f in r.findings} and "AS-306" in {p.rule_id for p in r.passed}
    # a mechanism found but not shown to reach every agent: a person says so, it is neither a gap nor a pass
    assert [f.status_hint for f in r.findings if f.rule_id == "AS-307"] == ["SIGN-OFF"]


def test_crewai_per_agent_bound_is_not_an_aggregate_budget(tmp_path: Path) -> None:
    obs = _obs(
        tmp_path,
        """
        from crewai import Agent
        a = Agent(role="A", goal="g", backstory="a", allow_delegation=True)
        b = Agent(role="B", goal="g", backstory="b")
    """,
    )
    assert "AS-306" in {f.rule_id for f in run(mode="audit", observed=obs).findings}


def test_a_gemini_id_with_no_version_marker_is_not_called_pinned() -> None:
    """Whether a bare Gemini id is fixed is a provider fact the id does not show: a person answers it."""
    assert pin_judge("gemini-2.5-flash")[0] == "undecidable"
    assert pin_judge("models/gemini-2.5-flash-image")[0] == "undecidable"
    assert not is_pinned("gemini-flash-latest") and not is_pinned("gemini-2.5-flash-latest")


# ---------------------------------------------------------------- multi-agent graph (cs-agents check)
HUB = """
from agents import Agent, function_tool, handoff
@function_tool
def lookup(q: str) -> str:
    \"\"\"Look up.\"\"\"
    return q
@function_tool
def refund(order: str) -> str:
    \"\"\"Refund.\"\"\"
    return order
def dynamic_instructions(ctx, agent):
    return "You are the Refunds Agent. Hand back to the Triage Agent when done."
triage = Agent(name="Triage Agent", instructions="Route requests.", tools=[lookup])
refunds = Agent(name="Refunds Agent", instructions=dynamic_instructions, tools=[refund])
triage.handoffs = [handoff(agent=refunds, on_handoff=None)]
refunds.handoffs.append(triage)
"""
HUB_SPEC = """
agent_id: triage
risk_level: medium
goal: {statement: route, done_criteria: [routed]}
out_of_scope: []
prohibited: []
approval_required: []
limits: []
escalation: []
untrusted_inputs: []
owner: o@example.com
agent_registry:
  - {id: triage, match: "Triage Agent", tools: [lookup]}
  - {id: refunds, match: "Refunds Agent", tools: [refund]}
declared_delegation: [triage -> refunds, refunds -> triage]
max_delegation_depth: 1
no_cycles: false
"""


def _hub(tmp: Path):
    from agenticreq.loader import load_spec_text

    (tmp / "agent.py").write_text(HUB)
    return build_observed(tmp), load_spec_text(HUB_SPEC)


def test_handoff_keyword_form_and_registry_by_name(tmp_path: Path) -> None:
    """handoff(agent=x) is read, and an agent whose prompt names another agent maps to itself."""
    from agenticreq.graph.analysis import build, resolve_registry

    obs, loaded = _hub(tmp_path)
    alias = resolve_registry(loaded.spec, obs)
    assert alias["Refunds Agent"] == "refunds" and alias["Triage Agent"] == "triage"
    g = build(obs, loaded.spec)
    assert {(e.src, e.dst) for e in g.edges} == {("triage", "refunds"), ("refunds", "triage")}
    r = run(mode="check", loaded=loaded, observed=obs)
    assert "AS-301" not in {f.rule_id for f in r.findings}


def test_hub_depth_is_measured_when_every_agent_is_a_target(tmp_path: Path) -> None:
    from agenticreq.graph.analysis import build

    obs, loaded = _hub(tmp_path)
    assert build(obs, loaded.spec).depth()[0] == 1  # simple paths only: triage -> refunds


def test_wire_prompts_of_one_agent_merge_by_tool_set(tmp_path: Path) -> None:
    """Runtime-built instructions give one agent several prompt hashes; its tools identify it."""
    from conftest import chat_call

    (tmp_path / "agent.py").write_text(HUB)
    tools = [{"name": "refund", "parameters": {"type": "object", "properties": {"order": {"type": "string"}}}}]
    calls = [
        chat_call(i, tools, system=f"You are the Refunds Agent, turn {i}. Ask the Triage Agent.") for i in range(3)
    ]
    obs = build_observed(tmp_path, captures=calls)
    assert [n.id for n in obs.nodes] == ["Refunds Agent"] and obs.nodes[0].calls == 3


# ---------------------------------------------------------------- AS-211 needs evidence (validation R2)
def _as211(tmp: Path, body: str, spec: str | None = None) -> bool:
    from agenticreq.loader import load_spec_text

    obs = _obs(
        tmp,
        "import uuid, random\nfrom langchain_core.tools import tool\n@tool\n"
        f'def book_seat(seat: str, state=None, store=None) -> str:\n    """Book."""\n{body}\n',
    )
    loaded = load_spec_text(spec) if spec else None
    return "AS-211" in {f.rule_id for f in run(mode="check" if spec else "audit", loaded=loaded, observed=obs).findings}


def test_as211_reuse_or_create_is_idempotent(tmp_path: Path) -> None:
    """cs-agents: `state.conf or random(...)`, stored back: a retry finds the first id."""
    assert not _as211(
        tmp_path, "    state.conf = state.conf or ''.join(random.choices('AB', k=6))\n    return state.conf"
    )


def test_as211_fresh_id_from_model_supplied_fallback(tmp_path: Path) -> None:
    """memory-agent: `memory_id or uuid4()` — the model omits memory_id on a retry: duplicate record."""
    assert _as211(tmp_path, "    mid = seat or uuid.uuid4()\n    store.aput('k', str(mid), {})\n    return ''")


def test_as211_name_only_is_silent_in_audit(tmp_path: Path) -> None:
    assert not _as211(tmp_path, "    state.seat = seat\n    return seat")


def test_as211_explicit_spec_requirement_fires_on_name(tmp_path: Path) -> None:
    spec = """
agent_id: a
risk_level: medium
goal: {statement: s, done_criteria: [d]}
out_of_scope: []
prohibited: []
approval_required: []
limits: []
escalation: []
untrusted_inputs: []
owner: o@example.com
write_safety: {idempotency_key: true}
"""
    assert _as211(tmp_path, "    state.seat = seat\n    return seat", spec)


def test_templates_only_ask_for_todos() -> None:
    """Spec templates: every problem is a TODO (AS-101), never a cascading type error (AS-107)."""
    from agenticreq.engine import validate_spec

    for t in (Path(__file__).parent.parent / "validation" / "templates").glob("*.yml"):
        ids = {f.rule_id for f in validate_spec(t).findings}
        assert ids == {"AS-101"}, (t.name, ids)


# ---------------------------------------------------------------- web-ui (browser-use) patterns
def test_browser_use_and_factory_patterns(tmp_path: Path) -> None:
    """browser-use/web-ui: @registry.action, a factory returning StructuredTool over partial(),
    library tool classes in a list, an Agent subclass, and test files left out."""
    files = {
        "controller.py": """
            from browser_use import Controller
            class MyController(Controller):
                def _register(self):
                    @self.registry.action("Upload a local file to the page")
                    async def upload_file(index: int, path: str, browser, available_file_paths: list[str]):
                        return None
        """,
        "research.py": """
            from functools import partial
            from langchain_core.tools import StructuredTool
            from langchain_community.tools.file_management import WriteFileTool, ReadFileTool
            from browser_use import Agent
            class BrowserUseAgent(Agent):
                pass
            async def _search(queries: list[str], task_id: str) -> list:
                return []
            def make_search(task_id):
                return StructuredTool.from_function(coroutine=partial(_search, task_id=task_id),
                                                    name="parallel_browser_search", description="Search.")
            def setup():
                tools = [WriteFileTool(), ReadFileTool()]
                tools += [make_search("t")]
                a = BrowserUseAgent(task="x")
                return tools
        """,
        "tests/test_agents.py": "from browser_use import Agent\nAgent(task='x', max_steps=100, llm=None)\n",
    }
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(dedent(src))
    obs = build_observed(tmp_path)
    assert {"upload_file", "parallel_browser_search", "write_file", "read_file"} <= set(obs.declared_tools)
    assert obs.declared_tools["upload_file"].description == "Upload a local file to the page"
    assert [p.name for p in obs.declared_tools["upload_file"].params] == ["index", "path"]
    assert obs.declared_tools["parallel_browser_search"].func_name == "_search"
    assert any(a.kind == "BrowserUseAgent" for a in obs.declared_agents)
    assert obs.framework == "browser-use"
    assert not any(b.code_ref and "tests/" in b.code_ref.file for b in obs.bounds)


def test_tool_groups_split_taint_contexts(tmp_path: Path) -> None:
    """web-ui: the research LLM's tools and the browser controller's actions are different contexts;
    a factory function's returned tool joins the list it is added to."""
    (tmp_path / "a.py").write_text(
        dedent("""
        from browser_use import Controller
        from langchain_core.tools import StructuredTool
        from langchain_community.tools.file_management import WriteFileTool
        class C(Controller):
            def reg(self):
                @self.registry.action("Upload a file")
                async def upload_file(index: int, path: str):
                    return None
        async def _web_search(queries: list[str]) -> list:
            return []
        def make_search():
            return StructuredTool.from_function(coroutine=_web_search, name="web_search", description="Search the web.")
        def setup(llm):
            tools = [WriteFileTool()]
            search = make_search()
            tools += [search]
            return llm.bind_tools(tools)
    """)
    )
    obs = build_observed(tmp_path)
    assert ["upload_file"] in obs.tool_groups and ["web_search", "write_file"] in obs.tool_groups
    paths = {tuple(f.path or []) for f in run(mode="audit", observed=obs).findings if f.rule_id == "AS-401"}
    assert ("web_search", "LLM", "write_file") in paths
    assert not any("upload_file" in p for p in paths)


# ---------------------------------------------------------------- needle (on-device assistant) patterns
def test_audit_gates_only_high_impact_writes(tmp_path: Path) -> None:
    """needle environments: a timer or a note needs no approval; replying to someone does."""
    obs = _obs(
        tmp_path,
        """
        import needle
        import requests
        @needle.tool
        def set_timer(minutes: int):
            \"\"\"Set a timer.\"\"\"
            return {}
        @needle.tool
        def create_note(text: str):
            \"\"\"Create a note.\"\"\"
            return {}
        @needle.tool
        def control_lights(room: str):
            \"\"\"Lights.\"\"\"
            home_store.set(room, "on")  # an actuator: writes (a store), not irreversible or external
            return {}
        @needle.tool
        def reply_to_notification(text: str):
            \"\"\"Reply.\"\"\"
            requests.post("https://device.example/reply", json={"text": text})  # sends: a write by its body
            return {}
        @needle.tool
        def find_my_phone():
            \"\"\"Ring the phone.\"\"\"
            return {}
        agent = needle.Needle(tools=[set_timer, create_note, control_lights, reply_to_notification, find_my_phone])
    """,
    )
    r = run(mode="audit", observed=obs)
    assert [f.title for f in r.findings if f.rule_id == "AS-212"] == [
        "Write tool reply_to_notification runs without approval"
    ]
    assert "AS-213" not in {f.rule_id for f in r.findings}  # find_my_phone() has nothing to search
    assert "AS-208" in {p.rule_id for p in r.passed}  # needle.run(max_steps=8)
    assert obs.framework == "needle"
    from agenticreq.ast.classify import is_write_tool

    assert is_write_tool(obs.declared_tools["control_lights"])  # actuator verbs are writes (taint sinks)


def test_check_at_high_risk_still_gates_every_write(tmp_path: Path) -> None:
    from agenticreq.loader import load_spec_text

    obs = _obs(
        tmp_path,
        'import needle, requests\n@needle.tool\ndef set_timer(minutes: int):\n    """T."""\n'
        "    requests.post('https://device.example/timer', json={'m': minutes})\n    return {}\n",
    )
    spec = load_spec_text("""
agent_id: a
risk_level: high
goal: {statement: s, done_criteria: [d]}
out_of_scope: []
prohibited: []
approval_required: []
limits: []
escalation: []
untrusted_inputs: []
owner: o@example.com
""")
    assert "AS-212" in {f.rule_id for f in run(mode="check", loaded=spec, observed=obs).findings}


# ---------------------------------------------------------------- klavis (MCP servers) patterns
def test_mcp_server_tools_are_not_agent_tools(tmp_path: Path) -> None:
    """klavis: an MCP server's tools are provided to clients. No agent context here, so no taint (AS-401)
    and no approval rule (AS-212); annotations decide read-only vs destructive."""
    obs = _obs(
        tmp_path,
        """
        from mcp import types
        from mcp.server import Server
        app = Server("docs")
        @app.list_tools()
        async def list_tools():
            return [
                types.Tool(name="get_document_info", description="Read a document.",
                           inputSchema={"type": "object", "properties": {"filename": {"type": "string"}},
                                        "required": ["filename"]},
                           annotations=types.ToolAnnotations(readOnlyHint=True)),
                types.Tool(name="delete_paragraph", description="Delete a paragraph.",
                           inputSchema={"type": "object", "properties": {"index": {"type": "integer"}}},
                           annotations=types.ToolAnnotations(destructiveHint=True, readOnlyHint=False)),
                types.Tool(name="send_email", description="Send.", inputSchema={"type": "object"}),
                types.Tool(name="delete_folder", description="Delete.", inputSchema={"type": "object"},
                           annotations=types.ToolAnnotations(**{"category": "X", "readOnlyHint": True})),
            ]
        token_url = "https://oauth2.example.com/token"
    """,
    )
    t = obs.declared_tools
    assert t["get_document_info"].kind == "mcp-server tool" and [p.name for p in t["get_document_info"].params] == [
        "filename"
    ]
    ids = [f.rule_id for f in run(mode="audit", observed=obs).findings]
    assert "AS-401" not in ids and "AS-212" not in ids and "AS-214" not in ids
    # only delete_folder claims read-only; unannotated send_email keeps MCP's destructive default
    assert [f.title for f in run(mode="audit", observed=obs).findings if f.rule_id == "AS-218"] == [
        "MCP tool delete_folder looks destructive but is annotated as safe"
    ]
    assert not {"AS-208", "AS-901", "AS-904"} & set(ids)  # a tool server has no agent loop to bound
    assert obs.framework == "mcp-server"


def test_klavis_blind_test_fixes(tmp_path: Path) -> None:
    """klavis blind test: token URLs are not secrets; a capped return is bounded; export / X_and_replace are
    writes; a FastMCP `@tool` alias marks server tools; server plumbing is not a hidden tool."""
    files = {
        "server.py": """
            from pathlib import Path
            from mcp.server.fastmcp import FastMCP
            mcp = FastMCP("s")
            tool = mcp.tool
            REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
            @tool()
            def list_objects(bucket: str) -> str:
                \"\"\"List objects.\"\"\"
                objects = client(bucket)
                return "\\n".join(f"- {o}" for o in objects[:20])
            @tool()
            def export_table(table: str, destination: str) -> str:
                \"\"\"Export a table.\"\"\"
                with open(destination, "w") as fh:  # writes: the body, not the name, makes it a write
                    fh.write(table)
                return "ok"
            @tool()
            def search_and_replace(filename: str, find: str, replace: str) -> str:
                \"\"\"Search and replace.\"\"\"
                path = Path(filename)
                path.write_text(path.read_text().replace(find, replace))
                return "ok"
            def run_streamable_http(port: int):
                return port
        """,
        "main.py": "from server import run_streamable_http\nrun_streamable_http(8000)\n",
    }
    for rel, src in files.items():
        (tmp_path / rel).write_text(dedent(src))
    obs = build_observed(tmp_path)
    from agenticreq.ast.classify import is_unbounded, is_write_tool

    t = obs.declared_tools
    assert "run_streamable_http" not in t
    assert all(x.kind == "mcp-server tool" for x in t.values())
    assert not is_unbounded(t["list_objects"])
    assert is_write_tool(t["export_table"]) and is_write_tool(t["search_and_replace"])
    assert not is_unbounded(t["search_and_replace"])
    assert not obs.secrets
