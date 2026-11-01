"""Validate the mock run on installed framework versions: one fixture agent per framework, run through `--python`.

    python validation/val_frameworks.py crewai=/path/to/venv/bin/python langgraph=/other/venv/bin/python ...

For each framework: the version installed in that environment, captured/total LLM calls, the tools the model
was offered, and whether the run proved anything (DiscoverResult.problems()). A version that passes here can
be added to `validated:` in src/agenticreq/data/supported_versions.yml.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agenticreq.capture import DiscoverError, discover

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
#: framework -> (fixture dir, distribution whose version is reported)
CASES = {
    "crewai": (FIX / "agents_crewai" / "crew_po", "crewai"),
    "langgraph": (FIX / "agents" / "langgraph_po", "langgraph"),
    "openai-agents": (FIX / "agents" / "oai_agents_po", "openai-agents"),
    "smolagents": (FIX / "agents" / "smol_tools", "smolagents"),
    "pydantic-ai": (FIX / "agents" / "pydantic_ai_po", "pydantic-ai-slim"),
    "autogen": (FIX / "agents" / "autogen_po", "autogen-agentchat"),
}


def _version(python: str, dist: str) -> str:
    out = subprocess.run([python, "-c", f"import importlib.metadata as m; print(m.version({dist!r}))"],
                         capture_output=True, text=True)
    return out.stdout.strip() or "not installed"


def main(argv: list[str]) -> int:
    rows, bad = [], 0
    for arg in argv:
        fw, _, python = arg.partition("=")
        d, dist = CASES[fw]
        row = {"framework": fw, "version": _version(python, dist), "python": python}
        try:
            r = discover(str(d / "agent.py"), mock_tools=str(d / "mocks.yml"), python=python,
                         timeout=240)
            cap, tot = r.coverage
            row |= {"coverage": f"{cap}/{tot}", "offered": sorted({t.name for c in r.calls for t in c.tools}),
                    "problems": r.problems(python), "notes": r.notes}
        except DiscoverError as e:
            row |= {"problems": [f"refused: {e}"]}
        bad += bool(row["problems"])
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(f"{len(rows) - bad}/{len(rows)} frameworks captured cleanly")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
