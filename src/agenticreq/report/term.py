"""Consoles that never break: a terminal without UTF-8 (Windows cp1252, a CI log, LANG=C) gets ASCII, and
one without colour gets plain text (Rich already honours NO_COLOR / TERM=dumb / a non-tty)."""

from __future__ import annotations

import codecs
import io
from typing import IO, Any

from rich import box
from rich.console import Console

_ASCII = str.maketrans(
    {
        "—": "-",
        "–": "-",
        "→": "->",
        "←": "<-",
        "↔": "<->",
        "…": "...",
        "✓": "ok",
        "✗": "x",
        "≤": "<=",
        "≥": ">=",
        "×": "x",
        "·": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "•": "*",
        "⚠": "!",
        "│": "|",
        "─": "-",
    }
)


def is_utf(stream: Any) -> bool:
    enc = getattr(stream, "encoding", None)
    if not enc:
        return False
    try:
        return codecs.lookup(enc).name.startswith("utf")
    except LookupError:
        return False


class AsciiFallback(io.TextIOBase):
    """A text stream that turns what the terminal cannot show into ASCII instead of raising."""

    def __init__(self, stream: IO[str]) -> None:
        self._s = stream
        self._enc = getattr(stream, "encoding", None) or "ascii"

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return self._enc

    def write(self, s: str) -> int:
        t = s.translate(_ASCII).encode(self._enc, "replace").decode(self._enc)
        self._s.write(t)
        return len(s)

    def flush(self) -> None:
        self._s.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._s, "isatty", lambda: False)())


def safe_stream(stream: IO[str]) -> IO[str]:
    return stream if is_utf(stream) else AsciiFallback(stream)  # type: ignore[return-value]


class SafeConsole(Console):
    """A Rich console whose stream is looked up on every write (like Rich's own default, so redirection and
    test runners still work) and wrapped in ASCII fallback when it cannot encode UTF-8."""

    @property
    def file(self) -> IO[str]:
        return safe_stream(super().file)

    @file.setter
    def file(self, new_file: IO[str]) -> None:
        self._file = new_file


def make_console(stderr: bool = False, file: IO[str] | None = None, **kw: Any) -> Console:
    return SafeConsole(stderr=stderr, file=file, highlight=False, **kw)


def panel_box(con: Console) -> box.Box:
    """ASCII borders when the console falls back to ASCII; a string buffer is written out as UTF-8 later."""
    return box.ASCII if isinstance(con.file, AsciiFallback) else box.ROUNDED
