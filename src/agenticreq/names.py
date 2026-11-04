"""One way to compare agent and tool names, everywhere a name is compared.

A framework's display name and the name sent to the model differ (CrewAI ``Web Search`` vs ``web_search``, a
LangChain ``search.web``, ``Search-Web``): they are one tool. ``norm`` lowercases, turns spaces, hyphens, dots (any
run of characters that is not a letter or a digit) into one underscore, and strips them from both ends. The
original spelling is kept for display: reports show names as the code spells them.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

_SEP = re.compile(r"[^a-z0-9]+")


@lru_cache(maxsize=1)
def _aliases() -> tuple[tuple[str, ...], dict[str, str]]:
    from agenticreq.data import load_yaml

    table = load_yaml("tool_aliases.yml")
    return tuple(table["prefixes"]), dict(table["aliases"])


def norm(name: str) -> str:
    """``Web Search`` -> ``web_search``, ``search.web`` -> ``search_web``, ``__Fetch--Page__`` -> ``fetch_page``. A
    provider's spelling of a hosted tool is the tool (``hosted:web_search``, ``web_search_20250305`` -> ``web_search``:
    data/tool_aliases.yml)."""
    prefixes, aliases = _aliases()
    for p in prefixes:
        name = name.removeprefix(p)
    text = _SEP.sub("_", name.lower()).strip("_")
    return aliases.get(text, text)


def same(a: str, b: str) -> bool:
    return norm(a) == norm(b)


def norm_set(names: Iterable[str]) -> set[str]:
    return {norm(n) for n in names}


def contains(names: Iterable[str], name: str) -> bool:
    """``name`` is in ``names`` however either is spelled."""
    return norm(name) in norm_set(names)


def not_in(names: Iterable[str], others: Iterable[str]) -> list[str]:
    """The names (as spelled in ``names``, in order, once each) that ``others`` does not have in any spelling."""
    have = norm_set(others)
    out: dict[str, str] = {}
    for n in names:
        if norm(n) not in have:
            out.setdefault(norm(n), n)
    return list(out.values())


def collisions(names: Iterable[str]) -> list[list[str]]:
    """Groups of different spellings that normalize to the same name (``save_file`` and ``Save File``): the code
    has two names the report cannot tell apart."""
    by: dict[str, list[str]] = {}
    for n in names:
        group = by.setdefault(norm(n), [])
        if n not in group:
            group.append(n)
    return [sorted(g) for _, g in sorted(by.items()) if len(g) > 1]
