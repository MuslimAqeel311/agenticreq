"""Onboarding contract.

1. installation creates nothing in the user's project
2. `import agenticreq` has no side effects
3. only `agenticreq init` creates spec.yml, and it never overwrites without --force
4. `agenticreq audit` works with zero configuration, no spec, and without the agent's dependencies
5. YAML is canonical; the Python builder API writes YAML
6. the README documents the journey: install -> audit -> init -> check
"""

from __future__ import annotations

import json
import os
import pkgutil
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import agenticreq
from agenticreq.cli import app

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures"
runner = CliRunner()


def snapshot(d: Path) -> set[str]:
    return {str(p.relative_to(d)) for p in d.rglob("*")}


# ------------------------------------------------------------------ 1. install creates nothing
def test_packaging_has_no_install_time_side_effects() -> None:
    tomllib = pytest.importorskip("tomllib")  # stdlib from Python 3.11

    assert not (ROOT / "setup.py").exists() and not (ROOT / "setup.cfg").exists()
    py = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert py["build-system"]["build-backend"] == "hatchling.build"
    hatch = py.get("tool", {}).get("hatch", {})
    assert "hooks" not in hatch.get("build", {}), "a build hook runs code at install"
    for target in hatch.get("build", {}).get("targets", {}).values():
        assert "hooks" not in target, "a build-target hook runs code at install"
    # a .pth file in the package would run code at every interpreter start
    assert not list((ROOT / "src").rglob("*.pth"))
    # wheels (what pip installs) cannot run code at install time; the only scripts are console entry points
    assert set(py["project"]["scripts"]) == {"agenticreq"}


# ------------------------------------------------------------------ 2. import has no side effects
def test_import_in_clean_cwd_writes_nothing(tmp_path: Path) -> None:
    work = tmp_path / "project"
    work.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    modules = sorted(
        m.name
        for m in pkgutil.walk_packages(agenticreq.__path__, "agenticreq.")
        if m.name != "agenticreq.capture.runner"
    )  # the runner is a subprocess entry point
    code = "import importlib, sys\nfor m in sys.argv[1:]:\n    importlib.import_module(m)\n"
    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }
    r = subprocess.run(
        [sys.executable, "-c", code, "agenticreq", *modules],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    assert snapshot(work) == set(), "import wrote into the current directory"
    assert snapshot(home) == set(), "import wrote into the home directory"


# ------------------------------------------------------------------ 3. only init creates spec.yml
@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of a real agent in a clean directory, which is also the cwd."""
    p = tmp_path / "my-agent"
    shutil.copytree(FIX / "agents" / "langgraph_po", p)
    monkeypatch.chdir(p)
    return p


def test_init_writes_only_the_spec(project: Path) -> None:
    before = snapshot(project)
    r = runner.invoke(app, ["init", ".", "--out", "spec.yml"])
    assert r.exit_code == 0, r.output
    assert snapshot(project) - before == {"spec.yml", "spec.schema.json"}  # the schema the first line points at


def test_init_refuses_to_overwrite_without_force(project: Path) -> None:
    (project / "spec.yml").write_text("mine: true\n", encoding="utf-8")
    r = runner.invoke(app, ["init", ".", "--out", "spec.yml"])
    assert r.exit_code == 2
    assert "refusing to overwrite" in " ".join(r.output.split())
    assert (project / "spec.yml").read_text(encoding="utf-8") == "mine: true\n"
    r = runner.invoke(app, ["init", ".", "--out", "spec.yml", "--force"])
    assert r.exit_code == 0, r.output
    assert yaml.safe_load((project / "spec.yml").read_text(encoding="utf-8"))["prohibited"] == "TODO"


def test_init_with_entry_leaves_no_capture_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("httpx")
    p = tmp_path / "agent"
    shutil.copytree(FIX / "agents" / "httpx_legacy", p)
    monkeypatch.chdir(p)
    before = snapshot(p)
    r = runner.invoke(app, ["init", ".", "--entry", "agent.py", "--mocks", "mocks.yml", "--out", "spec.yml"])
    assert r.exit_code == 0, r.output
    assert snapshot(p) - before == {"spec.yml", "spec.schema.json"}  # no .agenticreq/captures.json left behind
    d = yaml.safe_load((p / "spec.yml").read_text(encoding="utf-8"))["_derived"]
    assert d["observed_tools"] != "not seen yet"  # the captures were used, just not kept


@pytest.mark.parametrize(
    "cmd",
    [["audit", "."], ["audit", ".", "-f", "json"], ["stats", "."], ["rules"], ["params"], ["fix", "."], ["version"]],
)
def test_no_other_command_creates_files(project: Path, cmd: list[str]) -> None:
    before = snapshot(project)
    runner.invoke(app, cmd)
    assert snapshot(project) == before, f"`agenticreq {' '.join(cmd)}` wrote into the project"


# ------------------------------------------------------------------ 4. audit: zero config, AST only
def test_audit_zero_config_in_clean_project_without_agent_deps(tmp_path: Path) -> None:
    proj = tmp_path / "fresh"
    proj.mkdir()
    (proj / "agent.py").write_text(
        textwrap.dedent("""
        import definitely_not_an_installed_package as dep      # the agent's deps are NOT installed
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent

        @tool
        def send_payment(vendor_id: str, amount: float) -> str:
            '''Pay a vendor.'''
            return dep.pay(vendor_id, amount)

        agent = create_react_agent("gpt-4o", tools=[send_payment])
    """),
        encoding="utf-8",
    )
    before = snapshot(proj)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run(
        [sys.executable, "-m", "agenticreq.cli", "audit", ".", "-f", "json"],
        cwd=proj,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode in (0, 1), r.stderr  # findings decide the exit code; never a crash (2)
    data = json.loads(r.stdout)
    assert snapshot(proj) == before, "audit wrote into the project"
    ids = {f["rule_id"] for f in data["findings"]}
    assert {"AS-210", "AS-212"} <= ids  # it found real things with no spec at all
    # Level 1/3 fields are UNKNOWN, named, and point to init -> check; never implied to pass
    needs = next(f for f in data["findings"] if f["group"] == "mock_run" and f["rule_id"] == "AS-907")
    for field in ("prohibited", "limits", "approval_required", "escalation", "untrusted_inputs", "role_boundary"):
        assert field in needs["detail"]
    assert "agenticreq init" in needs["detail"] and "agenticreq check" in needs["detail"]
    assert not any(p["rule_id"] == "AS-907" for p in data.get("passed", []))


def test_audit_help_states_what_it_can_and_cannot_check() -> None:
    r = runner.invoke(app, ["audit", "--help"], terminal_width=200)
    out = " ".join(r.output.split())
    assert "CHECKED without a spec" in out and "NEEDS A SPEC" in out
    for item in (
        "model version pinned",
        "idempotency key",
        "hard-coded secrets",
        "fail-open",
        "self-modification",
        "environment separation",
        "structured output",
        "audit / trace hooks",
        "prohibited",
        "limits",
        "approval_required",
        "declared_delegation",
    ):
        assert item in out, item


# ------------------------------------------------------------------ 5. YAML canonical, Python emits YAML
SPEC_PY = """
from agenticreq.loader import SpecBuilder

SPEC = (SpecBuilder("po-agent").risk("high").goal("Raise POs", "PO created").owner("ops@x", "eng@x")
        .out_of_scope("pricing").prohibit("issue_refund", reason="finance only")
        .require_approval("submit_order", approver="procurement_lead").limit("submit_order", "amount", max=500000)
        .escalate("amount", ">", 500000, "cfo").untrusted("supplier_email").allow("submit_order", "get_inventory")
        .data_rules(never_store=["card_number"]))
"""


def test_builder_writes_yaml_that_check_reads(tmp_path: Path) -> None:
    from agenticreq.loader import SpecBuilder, load_spec

    ns: dict[str, object] = {}
    exec(SPEC_PY, ns)  # noqa: S102 - test fixture
    b = ns["SPEC"]
    assert isinstance(b, SpecBuilder)
    out = b.write(tmp_path / "spec.yml")
    loaded = load_spec(out)
    assert loaded.spec is not None and loaded.spec.agent_id == "po-agent"
    assert loaded.spec.prohibited_tools() == {"issue_refund"}
    with pytest.raises(FileExistsError):
        b.write(out)


def test_export_spec_cli(tmp_path: Path) -> None:
    from agenticreq.loader import load_spec

    (tmp_path / "my_spec.py").write_text(SPEC_PY, encoding="utf-8")
    out = tmp_path / "spec.yml"
    r = runner.invoke(app, ["export-spec", f"{tmp_path / 'my_spec.py'}:SPEC", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert load_spec(out).spec is not None
    r = runner.invoke(app, ["export-spec", f"{tmp_path / 'my_spec.py'}:SPEC", "--out", str(out)])
    assert r.exit_code == 2 and "refusing to overwrite" in " ".join(r.output.split())


def test_agentspec_round_trips_through_yaml(tmp_path: Path) -> None:
    from agenticreq.loader import load_spec_text, spec_to_yaml

    text = (FIX / "specs" / "good.yml").read_text(encoding="utf-8")
    first = load_spec_text(text).spec
    assert first is not None
    again = load_spec_text(spec_to_yaml(first)).spec
    assert again is not None
    assert again.model_dump(exclude={"derived"}) == first.model_dump(exclude={"derived"})


# ------------------------------------------------------------------ 6. README journey
def test_readme_documents_the_journey_in_order() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index("## Quick start")
    section = text[start : text.index("\n## ", start + 5)]
    steps = ["pip install", "agenticreq audit", "agenticreq init", "agenticreq check"]
    pos = [section.index(s) for s in steps]
    assert pos == sorted(pos), "README must show install -> audit -> init -> check, in that order"
    assert "discover" not in section[: pos[2]], "value first: no capture setup before init"
