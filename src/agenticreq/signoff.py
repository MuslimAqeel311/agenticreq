"""What an entry under ``confirmed:`` in spec.yml counts as.

Each entry answers one sign-off item: ``answer: yes`` (it is in place: SIGNED), ``no`` (it is not in place yet: a
gap), or ``not_needed`` (it does not apply here: done). An entry missing a field its answer needs stays an open
sign-off ("evidence is missing"). A signed item goes back to an open sign-off when its ``expires`` date has passed,
or when a file named in its ``evidence`` (``path/to/file.py:40``) has changed since the sign-off ``date``: what the
person confirmed may no longer be what the project does.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from agenticreq.models.spec import Attestation

#: a file:line reference in free text ("graph.py:40", "src/app/flags.py:12")
_FILE_REF = re.compile(r"(?<![\w/\\.-])((?:[\w.-]+[/\\])*[\w.-]+\.[A-Za-z0-9]+):(\d+)")


def evidence_files(evidence: str | None) -> list[str]:
    """The files a sign-off's evidence points at, in order."""
    return list(dict.fromkeys(m.group(1) for m in _FILE_REF.finditer(evidence or "")))


#: the fields each answer needs, in the order they are reported missing
REQUIRED: dict[str, tuple[str, ...]] = {
    "yes": ("by", "date", "expires", "evidence"),
    "no": ("by", "date", "reason"),
    "not_needed": ("by", "date", "reason"),
}

State = Literal["signed", "open", "no", "not_needed"]


@dataclass(frozen=True)
class Assessment:
    """``state``: signed (yes, complete, current) | open (a field missing, expired, evidence changed) | no |
    not_needed. ``message``: why it is open, or the person's reason for no / not_needed."""

    state: State
    message: str = ""


def missing(a: Attestation) -> list[str]:
    return [f for f in REQUIRED[a.answer] if not getattr(a, f)]


def lapsed(a: Attestation, today: dt.date, root: str | Path | None = None) -> str:
    """Why this sign-off no longer counts ("sign-off expired on ...", "...: file changed since sign-off ..."),
    or "" when it still does."""
    if a.expires and a.expires < today:
        return f"sign-off expired on {a.expires}"
    if a.date is None:
        return ""
    bases = [Path(root)] if root is not None else []
    bases.append(Path.cwd())
    for name in evidence_files(a.evidence):
        for base in bases:
            p = base / name
            if p.is_file():
                changed = _last_change(p)
                if changed > a.date:
                    return f"{name}: file changed since sign-off ({changed}, signed {a.date})"
                break
    return ""


def _last_change(p: Path) -> dt.date:
    """When the file's content last changed: its last git commit when it is committed and unchanged since (a
    checkout or a copy touches the mtime, not the content), else its modification time."""
    import subprocess

    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", p.name],
            cwd=p.parent,
            capture_output=True,
            encoding="utf-8",
            text=True,
            timeout=5,
        )
        log = subprocess.run(
            ["git", "log", "-1", "--format=%cs", "--", p.name],
            cwd=p.parent,
            capture_output=True,
            encoding="utf-8",
            text=True,
            timeout=5,
        )
        if dirty.returncode == 0 and not dirty.stdout.strip() and log.returncode == 0 and log.stdout.strip():
            return dt.date.fromisoformat(log.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return dt.date.fromtimestamp(p.stat().st_mtime)


def assess(a: Attestation, today: dt.date, root: str | Path | None = None) -> Assessment:
    gaps = missing(a)
    if gaps:
        return Assessment("open", f"{' and '.join(gaps)} {'is' if len(gaps) == 1 else 'are'} missing")
    why = lapsed(a, today, root)
    if why:
        return Assessment("open", why)
    if a.answer == "yes":
        return Assessment("signed")
    return Assessment(a.answer, a.reason or "")


def counts(a: Attestation, today: dt.date, root: str | Path | None = None) -> bool:
    """The entry confirms its item: answered yes, every field there, not expired, evidence unchanged."""
    return assess(a, today, root).state == "signed"
