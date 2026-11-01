"""A lower layer must not import a higher one (see docs/architecture.md).

Layers, lowest first. Imports may point down or sideways, never up:

    0 foundation  models, defaults, data, names, pathutil, versions, jsonschema_util, shapes, signoff, subfields,
                  not_needed, params, thresholds, pyenv
    1 loader      spec.yml -> typed spec
    2 analysis    ast, detectors, recognizers, capture, traces, mcp, graph, observe, export, fix
    3 rules
    4 engine      engine, derived, coverage_table, progress
    5 report
    6 cli         cli, scaffold, mockgen, schema

``_KNOWN_UPWARD`` lists the upward imports that exist today (each is a function-level import that breaks a cycle).
The list may only shrink: a new upward import fails this test, and so does an entry that no longer exists.
"""

from __future__ import annotations

import ast
from pathlib import Path

import agenticreq

_ROOT = Path(agenticreq.__file__).resolve().parent

_LAYER: dict[str, int] = {}
for _layer, _names in enumerate(
    [
        "models defaults data names pathutil versions jsonschema_util shapes signoff subfields not_needed params "
        "thresholds pyenv __init__ __version__",
        "loader",
        "ast detectors recognizers capture traces mcp graph observe export fix",
        "rules",
        "engine derived coverage_table progress",
        "report",
        "cli scaffold mockgen schema",
    ]
):
    for _n in _names.split():
        _LAYER[_n] = _layer

_KNOWN_UPWARD = {
    ("data", "ast"),
    ("loader", "capture"),
    ("loader", "rules"),
    ("models", "ast"),
    ("models", "rules"),
    ("pyenv", "ast"),
    ("thresholds", "rules"),
    ("traces", "rules"),
}


def _component(dotted: str) -> str:
    parts = dotted.split(".")
    return parts[1] if len(parts) > 1 else "__init__"


def _imports() -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for f in _ROOT.rglob("*.py"):
        rel = f.relative_to(_ROOT).with_suffix("")
        parts = list(rel.parts)
        source = parts[0]
        pkg = ["agenticreq", *parts[:-1]]
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            mods: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg[: len(pkg) - (node.level - 1)]
                    mod = ".".join([*base, *([node.module] if node.module else [])])
                else:
                    mod = node.module or ""
                if mod.split(".")[0] == "agenticreq":
                    mods.append(mod)
                    if mod == "agenticreq":
                        mods.extend(f"agenticreq.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                mods.extend(a.name for a in node.names if a.name.split(".")[0] == "agenticreq")
            for m in mods:
                target = _component(m)
                if target != source:
                    edges.add((source, target))
    return edges


def test_every_top_level_module_has_a_layer() -> None:
    names = {p.stem if p.is_file() else p.name for p in _ROOT.iterdir() if (p.suffix == ".py" or p.is_dir())}
    names.discard("__pycache__")
    assert names <= set(_LAYER), (
        f"add these to the layer table in tests/test_layering.py: {sorted(names - set(_LAYER))}"
    )


def test_no_new_upward_import() -> None:
    upward = {(a, b) for a, b in _imports() if _LAYER[b] > _LAYER[a]}
    new = upward - _KNOWN_UPWARD
    assert not new, f"a lower layer imports a higher one: {sorted(new)} (see docs/architecture.md)"


def test_known_upward_list_has_no_stale_entry() -> None:
    upward = {(a, b) for a, b in _imports() if _LAYER[b] > _LAYER[a]}
    assert _KNOWN_UPWARD <= upward, f"remove from _KNOWN_UPWARD: {sorted(_KNOWN_UPWARD - upward)}"
