"""Shipped data files (allowlists). Loaded via importlib.resources-free path lookup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DATA_DIR = Path(__file__).resolve().parent


def load_yaml(name: str) -> Any:
    return yaml.safe_load((DATA_DIR / name).read_text(encoding="utf-8"))


class InjectedTools:
    def __init__(self, extra: list[str] | None = None) -> None:
        from agenticreq.ast.classify import canonical

        data = load_yaml("known_injected_tools.yml") or {}
        self.exact: dict[str, str] = {}
        self.prefix: list[tuple[str, str]] = []
        for fw, body in (data.get("frameworks") or {}).items():
            for n in body.get("exact") or []:
                self.exact[canonical(n)] = fw
            for p in body.get("prefix") or []:
                self.prefix.append((canonical(p) + ("_" if p.endswith("_") else ""), fw))
        for n in extra or []:
            self.exact[canonical(n)] = "user"

    def framework_of(self, name: str) -> str | None:
        from agenticreq.ast.classify import canonical

        c = canonical(name)
        if c in self.exact:
            return self.exact[c]
        for p, fw in self.prefix:
            if c.startswith(p):
                return fw
        return None
