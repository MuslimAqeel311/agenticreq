"""Where an agent seen only at run time comes from, when no declared agent in the code matches it.

A model call whose system prompt matches no declared agent is either made by the framework itself (a converter that
formats the final answer as JSON, a planner, a summarizer: framework text around the project's data) or by the
project's own code in a way the code does not spell out (an agent built from a config file, a factory). The test is
framework-agnostic: the project's own agents carry text written in the project (a role, a backstory, instructions
in a .py, .yaml or .md file) at the start of their prompt, or offer the project's tools. A call that does neither
was made by the framework.

Generated ids (a prompt hash such as ``a5119f``, a uuid, ``Agent#2``) are never names a person can register.
"""

from __future__ import annotations

import ast
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from agenticreq.ast.classify import canonical

if TYPE_CHECKING:
    from agenticreq.models.observed import ObservedAgent

#: how much of a prompt is the agent's own opening (a role, a backstory): project text deep inside a framework
#: prompt (a field description in a JSON schema) does not make the call the project's
OPENING = 400
#: the shortest project text that counts (shorter strings, "Approved", "0.0 - 1.0", match by chance)
MIN_TEXT = 24
_TEXT_FILES = (".yaml", ".yml", ".md", ".txt", ".json", ".toml", ".prompt", ".j2", ".jinja", ".jinja2")
_SKIP_DIRS = frozenset(
    {
        ".git",
        ".agenticreq",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "site-packages",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".tox",
    }
)
_GENERATED = re.compile(
    r"^(?:[0-9a-f]{6,64}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|"
    r".*#\d+|(?:agent|node|tool|step|task)[_-]?\d+|(?:agent|node)[_-][0-9a-f]{6,})$",
    re.I,
)


def looks_generated(name: str | None) -> bool:
    """A hash, a uuid or a generated id (``a5119f``, ``Agent#2``, ``agent_3``): not a name in the code."""
    return not name or bool(_GENERATED.match(name.strip()))


def _squash(text: str) -> str:
    return " ".join(text.split())


def _py_texts(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError, OSError):
        return []
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _file_texts(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for line in text.splitlines():  # a YAML value, a Markdown paragraph: the text after a `key:`
        line = re.sub(r"^\s*(?:[-*>]\s*)?(?:[\w .-]+:\s*)?[|>]?[-+]?\s*", "", line).strip(" \"'")
        out.append(line)
    return out


@lru_cache(maxsize=16)
def _project_fragments(root: str) -> frozenset[str]:
    """The opening of every string written in the project (code literals, config and prompt files)."""
    out: set[str] = set()
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
        for f in files:
            p = Path(dirpath) / f
            texts = _py_texts(p) if f.endswith(".py") else _file_texts(p) if f.endswith(_TEXT_FILES) else []
            for t in texts:
                for piece in re.split(r"\{[^{}]*\}", t):  # a template's fixed text, not its {placeholders}
                    s = _squash(piece)
                    if len(s) >= MIN_TEXT:
                        out.add(s[: MIN_TEXT * 2])
    return frozenset(out)


def project_wrote(obs: ObservedAgent, prompt: str | None) -> bool:
    """The opening of ``prompt`` holds text written in the project."""
    if not prompt:
        return False
    head = _squash(prompt)[:OPENING]
    return any(f in head for f in _project_fragments(str(Path(obs.root).resolve())))


def origin(obs: ObservedAgent, prompt: str | None, tools: list[str]) -> str:
    """``framework`` (the framework made this call itself) or ``unresolved`` (the project's own agent, its name not
    found in the code) for a runtime agent no declared agent matches."""
    from agenticreq.data import InjectedTools

    if not obs.files:  # no project code to compare with (a captures-only check): it cannot be told apart
        return "unresolved"
    injected = InjectedTools()
    declared = {canonical(t) for t in obs.declared_tools}
    if any(canonical(t) in declared and not injected.framework_of(t) for t in tools):
        return "unresolved"
    return "unresolved" if project_wrote(obs, prompt) else "framework"
