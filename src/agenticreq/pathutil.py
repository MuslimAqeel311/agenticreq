"""Path comparisons that hold on every OS: Windows and macOS (APFS default) compare names case-insensitively,
so `Agent/tools.py` and `agent/tools.py` are one file there and two on Linux."""

from __future__ import annotations

import sys
from pathlib import Path, PurePath

CASE_INSENSITIVE = sys.platform in ("win32", "darwin")


def _key(p: PurePath) -> tuple[str, ...]:
    return tuple(x.casefold() for x in p.parts) if CASE_INSENSITIVE else p.parts


def relative(p: Path, base: Path) -> Path | None:
    """`p` relative to `base` (keeping p's own spelling), or None when p is not inside base."""
    if CASE_INSENSITIVE:  # the OS already compares case-insensitively, so pathlib's own check is right
        try:
            return p.relative_to(base)
        except ValueError:
            pass
    kp, kb = _key(p), _key(base)
    if len(kp) >= len(kb) and kp[: len(kb)] == kb:
        return Path(*p.parts[len(kb) :])
    return None


def is_within(p: Path, base: Path) -> bool:
    return relative(p, base) is not None


def same_path(a: Path, b: Path) -> bool:
    return _key(a) == _key(b)


def path_key(p: PurePath) -> str:
    """A string that is equal for two spellings of one file (for cycle / duplicate detection)."""
    return "/".join(_key(p))
