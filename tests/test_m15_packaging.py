"""M15: Windows, macOS and Linux classifiers, httpx bounded below 1 so a pre-release cannot break capture, and the
docs linked from --help. Name, version scheme and entry point are unchanged."""

from __future__ import annotations

import sys
from pathlib import Path

from typer.testing import CliRunner

from agenticreq.cli import app

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

PYPROJECT = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text(encoding="utf-8"))


def test_m15_classifiers_name_all_three_oses() -> None:
    cl = PYPROJECT["project"]["classifiers"]
    assert {
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
    } <= set(cl)


def test_m15_httpx_is_bounded_below_1() -> None:
    opt = PYPROJECT["project"]["optional-dependencies"]
    assert "httpx>=0.28,<1" in opt["capture"] and "httpx>=0.28,<1" in opt["test"]


def test_m15_name_version_and_entry_point_unchanged() -> None:
    assert PYPROJECT["project"]["name"] == "agenticreq"
    assert PYPROJECT["project"]["dynamic"] == ["version"]
    assert PYPROJECT["project"]["scripts"]["agenticreq"] == "agenticreq.cli:app"


def test_m15_help_links_the_docs() -> None:
    out = " ".join(CliRunner().invoke(app, ["--help"], terminal_width=200).output.split())
    assert "github.com/MuslimAqeel311/agenticreq/tree/main/docs" in out
