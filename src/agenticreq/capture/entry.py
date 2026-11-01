"""Entry-point parsing shared by `discover` (parent) and the capture runner (child)."""

from __future__ import annotations


def split_entry(entry: str) -> tuple[str, str]:
    """Split ``target[:func]`` into (target, func).

    Splits on the *last* colon and only when what follows is a Python identifier, so a Windows
    drive letter (``C:\\agents\\agent.py``) is never mistaken for a ``module:func`` separator.
    """
    head, sep, tail = entry.rpartition(":")
    if sep and head and tail.isidentifier():
        return head, tail
    return entry, ""
