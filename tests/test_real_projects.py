"""The user's real projects, re-checked after every change (skipped where a folder is not on this machine). Each
must give a report with no crash, one line per parameter, nothing hidden or cut, the project's own framework named,
and no gap asking to register a generated id."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

D = Path(r"C:\Users\muslim.aqeel\Downloads")
PROJECTS = {
    "agentic_intel_crew": D / "agentic_intel_crew" / "agentic_intel_crew",
    "docmind": D / "docmind" / "docmind",
    "finance": D / "practice_single" / "finance",
    "autogen_research_studio": D / "autogen-research-studio" / "autogen-research-studio",
    "dev_agent_team": D / "dev_agent_team" / "dev_agent_team",
    "single_agent_analyst": D / "single_agent_analyst" / "single_agent_analyst",
}
SRC = str(Path(__file__).resolve().parent.parent / "src")


def _check(d: Path) -> dict:  # type: ignore[type-arg]
    env = {**os.environ, "PYTHONPATH": SRC, "AGENTICREQ_NO_STATE": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(
        [sys.executable, "-c", "from agenticreq.cli import app; app()", "check", ".", "-f", "json", "--quiet"],
        cwd=d,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert r.stdout.strip(), r.stderr[-800:]
    return json.loads(r.stdout)  # type: ignore[no-any-return]


@pytest.mark.parametrize("name", list(PROJECTS))
def test_real_project(name: str) -> None:
    d = PROJECTS[name]
    if not (d / "spec.yml").is_file():
        pytest.skip(f"{name} is not on this machine")
    data = _check(d)
    lines = data["results"]
    keys = [(r["agent"], r["param"]) for r in lines]
    assert len(keys) == len(set(keys)), "a parameter in two lines"
    fw = (data.get("metadata", {}).get("framework") or "").lower()
    for r in lines:
        t = r["reason"]
        assert "more)" not in t and "…" not in t, t
        if fw and "openai" not in fw:
            assert "(openai-agents)" not in t, t
        if r["status"] == "GAP":
            assert not re.search(r"\b(?=[0-9a-f]*\d)[0-9a-f]{6,}\b", t), t
