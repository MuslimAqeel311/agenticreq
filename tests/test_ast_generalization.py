"""AST extraction across the ways real projects turn functions into tools, and across file layouts.

Each case writes a tiny project to tmp_path and checks declared / bound tools. Nothing is imported:
the frameworks named in the sources are not installed for these tests.
"""

from pathlib import Path
from textwrap import dedent

import pytest

from agenticreq.observe import build_observed

TOOLS_PY = '''
def lookup(sku: str) -> dict:
    """Look up stock.

    Args:
        sku: Product code.
    """
    return {}


def pay(invoice_id: str, amount: float) -> dict:
    """Pay an invoice."""
    if amount > 5000:
        raise ValueError
    return {}


def helper(x):
    return x
'''


def _project(tmp: Path, files: dict[str, str]) -> Path:
    for rel, src in files.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(dedent(src))
    return tmp


CASES = {
    "langchain curried wrapper over imported fns": {
        "tools.py": TOOLS_PY,
        "agent.py": """
            from langchain_core.tools import tool
            from langgraph.prebuilt import create_react_agent
            from tools import lookup, pay
            def main():
                tools = [tool(parse_docstring=True)(f) for f in (lookup, pay)]
                return create_react_agent("m", tools)
        """,
    },
    "module alias attribute access": {
        "tools.py": TOOLS_PY,
        "agent.py": """
            import tools as t
            from pydantic_ai import Agent
            agent = Agent("openai:gpt-4o-2024-08-06", tools=[t.lookup, t.pay])
        """,
    },
    "package with relative import and re-export": {
        "app/__init__.py": "",
        "app/tools/__init__.py": "from .impl import lookup, pay\n",
        "app/tools/impl.py": TOOLS_PY,
        "app/agent.py": """
            from agents import Agent, function_tool
            from .tools import lookup, pay
            a = Agent(name="x", tools=[function_tool(lookup), function_tool(pay)])
        """,
    },
    "tool list constant in another module": {
        "tools.py": TOOLS_PY,
        "registry.py": "from tools import lookup, pay\nTOOLS = [lookup, pay]\n",
        "agent.py": """
            from smolagents import ToolCallingAgent, tool
            from registry import TOOLS
            agent = ToolCallingAgent(tools=[tool(f) for f in TOOLS], model=None)
        """,
    },
    "StructuredTool.from_function with func= kw": {
        "tools.py": TOOLS_PY,
        "agent.py": """
            from langchain_core.tools import StructuredTool
            from tools import lookup, pay
            t1 = StructuredTool.from_function(func=lookup)
            t2 = StructuredTool.from_function(func=pay, name="pay")
            llm = None
            bound = llm.bind_tools([t1, t2])
        """,
    },
    "LlamaIndex FunctionTool.from_defaults(fn=)": {
        "tools.py": TOOLS_PY,
        "agent.py": """
            from llama_index.core.tools import FunctionTool
            from llama_index.core.agent import ReActAgent
            import tools
            agent = ReActAgent(tools=[FunctionTool.from_defaults(fn=tools.lookup),
                                      FunctionTool.from_defaults(fn=tools.pay)])
        """,
    },
    "Google ADK / Agno bare functions in tools=": {
        "tools.py": TOOLS_PY,
        "agent.py": """
            from google.adk.agents import LlmAgent
            from tools import lookup, pay
            root = LlmAgent(name="root", model="gemini-2.0-flash", tools=[lookup, pay])
        """,
    },
    "AutoGen register_function": {
        "tools.py": TOOLS_PY,
        "agent.py": """
            from autogen import register_function, AssistantAgent
            from tools import lookup, pay
            a = AssistantAgent("a")
            register_function(lookup, caller=a, executor=a, description="lookup")
            register_function(pay, caller=a, executor=a, description="pay")
        """,
    },
    "tool class inheriting a project base class": {
        "base.py": "from crewai.tools import BaseTool\nclass MyBase(BaseTool):\n    pass\n",
        "tools.py": """
            from base import MyBase
            class LookupTool(MyBase):
                name: str = "lookup"
                description: str = "Look up stock."
                def _run(self, sku: str) -> str:
                    return ""
            class PayTool(MyBase):
                name: str = "pay"
                description: str = "Pay."
                def _run(self, invoice_id: str, amount: float) -> str:
                    return ""
        """,
        "agent.py": """
            from crewai import Agent
            from tools import LookupTool, PayTool
            a = Agent(role="r", goal="g", backstory="b", tools=[LookupTool(), PayTool()])
        """,
    },
    "list concatenation and starred": {
        "tools.py": TOOLS_PY,
        "agent.py": """
            from langchain_core.tools import tool
            from langgraph.prebuilt import create_react_agent
            from tools import lookup, pay
            READ = [tool(lookup)]
            WRITE = [tool(pay)]
            agent = create_react_agent("m", tools=[*READ] + WRITE)
        """,
    },
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_tools_found_and_bound(tmp_path: Path, case: str) -> None:
    root = _project(tmp_path, CASES[case])
    obs = build_observed(root)
    assert {"lookup", "pay"} <= set(obs.declared_tools), case
    assert {"lookup", "pay"} <= set(obs.bound_tools), case
    assert "helper" not in obs.declared_tools  # a plain function nobody binds is not a tool


def test_single_entry_file_pulls_in_imported_modules(tmp_path: Path) -> None:
    root = _project(tmp_path, CASES["langchain curried wrapper over imported fns"])
    obs = build_observed(root / "agent.py")
    assert set(obs.declared_tools) == {"lookup", "pay"}
    pay = obs.declared_tools["pay"]
    assert pay.code_ref and pay.code_ref.file == "tools.py"
    assert pay.body_bounds == {"amount": [5000.0]}  # body facts come from the real definition
    assert [p.name for p in obs.declared_tools["lookup"].params] == ["sku"]
    assert obs.declared_tools["lookup"].params[0].description == "Product code."


def test_unbound_function_list_is_not_tools(tmp_path: Path) -> None:
    root = _project(
        tmp_path, {"tools.py": TOOLS_PY, "agent.py": "from tools import lookup, pay\nHANDLERS = [lookup, pay]\n"}
    )
    assert build_observed(root).declared_tools == {}


def test_third_party_imports_are_not_followed(tmp_path: Path) -> None:
    root = _project(tmp_path, {"agent.py": "from langchain_core.tools import tool\nimport json\n"})
    obs = build_observed(root / "agent.py")
    assert obs.files == ["agent.py"]


def test_import_cycle_terminates(tmp_path: Path) -> None:
    root = _project(tmp_path, {"a.py": "from b import x\ny = 1\n", "b.py": "from a import y\nx = y\n"})
    assert build_observed(root / "a.py").files == ["a.py", "b.py"]


def test_name_override_and_crewai_string_name(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        {
            "tools.py": TOOLS_PY,
            "agent.py": """
        from agents import function_tool
        from crewai.tools import tool
        from tools import lookup, pay
        a = function_tool(lookup, name_override="stock_lookup")
        b = tool("Pay Invoice")(pay)
    """,
        },
    )
    assert {"stock_lookup", "Pay Invoice"} <= set(build_observed(root).declared_tools)


def test_prompt_built_from_imported_constants(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        {
            "prompts.py": 'RULES = "Never pay above 5000."\n',
            "agent.py": """
        from agents import Agent
        from prompts import RULES
        a = Agent(name="Payer", instructions="You pay invoices. " + RULES)
        b = Agent(name="Other", instructions=f"Other agent. {RULES}")
    """,
        },
    )
    prompts = {a.name: a.system_prompt for a in build_observed(root).declared_agents}
    assert prompts == {
        "Payer": "You pay invoices. Never pay above 5000.",
        "Other": "Other agent. Never pay above 5000.",
    }


def test_direct_call_of_tool_module_function(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        {
            "tools.py": TOOLS_PY,
            "agent.py": """
        from langchain_core.tools import tool
        from tools import helper, lookup, pay
        t = tool(pay)
        h = tool(helper)
        def prefetch():
            return lookup("A")
    """,
        },
    )
    obs = build_observed(root / "agent.py")
    assert "lookup" in obs.direct_calls
    assert obs.declared_tools["lookup"].kind == "tool-module function (code pattern)"


def test_helper_in_utils_module_is_not_a_tool(tmp_path: Path) -> None:
    """Only 1 of 3 functions is a tool: a utils module, so calling a helper is not a hidden tool."""
    root = _project(
        tmp_path,
        {
            "tools.py": TOOLS_PY,
            "agent.py": """
        from langchain_core.tools import tool
        from tools import helper, pay
        t = tool(pay)
        x = helper(1)
    """,
        },
    )
    obs = build_observed(root / "agent.py")
    assert set(obs.declared_tools) == {"pay"} and obs.direct_calls == {}


def test_agent_as_tool_and_schema_class(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        {
            "state.py": """
        from pydantic import BaseModel, Field
        class ConductResearch(BaseModel):
            \"\"\"Research a topic.\"\"\"
            topic: str = Field(description="The topic.")
        class Done(BaseModel):
            \"\"\"Finished.\"\"\"
        class NotATool(BaseModel):
            x: int
    """,
            "agent.py": """
        from agents import Agent
        from state import ConductResearch, Done, NotATool
        analyst = Agent(name="Analyst", instructions="a")
        boss = Agent(name="Boss", tools=[analyst.as_tool(tool_name="analyse", tool_description="d")])
        def node(llm):
            tools = [ConductResearch, Done]
            return llm.bind_tools(tools), NotATool(x=1)
    """,
        },
    )
    obs = build_observed(root / "agent.py")
    assert {"analyse", "ConductResearch", "Done"} <= set(obs.bound_tools)
    assert "NotATool" not in obs.declared_tools
    assert obs.declared_tools["ConductResearch"].params[0].description == "The topic."


def test_framework_source_inside_repo_is_not_followed(tmp_path: Path) -> None:
    """An example inside a framework's own repo must not inventory the framework's built-in tools."""
    root = _project(
        tmp_path,
        {
            "pyproject.toml": "",
            "src/smolagents/__init__.py": "from .default_tools import FinalAnswerTool\n",
            "src/smolagents/default_tools.py": "class Tool: ...\nclass FinalAnswerTool(Tool):\n    name = 'final_answer'\n",
            "examples/demo/run.py": "from smolagents import CodeAgent, tool\n@tool\ndef mine(x: int) -> int:\n"
            '    """Mine."""\n    return x\nagent = CodeAgent(tools=[mine], model=None)\n',
        },
    )
    obs = build_observed(root / "examples" / "demo")
    assert set(obs.declared_tools) == {"mine"}


def test_late_handoffs_and_tools(tmp_path: Path) -> None:
    """cs-agents style: handoffs and tools wired after the agents are constructed (validation V9)."""
    root = _project(
        tmp_path,
        {
            "tools.py": TOOLS_PY,
            "agent.py": """
        from agents import Agent, function_tool
        from tools import lookup, pay
        triage = Agent(name="Triage")
        billing = Agent(name="Billing")
        triage.handoffs = [billing]
        billing.tools.append(function_tool(pay))
        billing.tools.extend([function_tool(lookup)])
    """,
        },
    )
    obs = build_observed(root)
    agents = {a.name: a for a in obs.declared_agents}
    assert agents["Triage"].handoffs == ["billing"]
    assert set(agents["Billing"].tools) == {"pay", "lookup"}
    from agenticreq.graph.builder import declared_edges

    assert ("Triage", "Billing", "handoffs=") in declared_edges(obs)


def test_model_in_config_field(tmp_path: Path) -> None:
    """open_deep_research style config (validation V6)."""
    root = _project(
        tmp_path,
        {
            "config.py": """
        from pydantic import BaseModel, Field
        class Configuration(BaseModel):
            research_model: str = Field(default="openai:gpt-4.1")
            summarization_model: str = "openai:gpt-4.1-mini-2025-04-14"
            max_steps: int = Field(default=5)
    """
        },
    )
    models = {m.name: m.pinned for m in build_observed(root).models}
    assert models == {"openai:gpt-4.1": False, "openai:gpt-4.1-mini-2025-04-14": True}
