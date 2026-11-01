"""spec.schema.json (repo root and inside the package) matches the spec model."""

from __future__ import annotations

import json
from pathlib import Path

import agenticreq
from agenticreq.schema import SCHEMA_FILE, spec_json_schema

_PKG = Path(agenticreq.__file__).resolve().parent


def test_packaged_schema_is_current() -> None:
    packaged = json.loads((_PKG / SCHEMA_FILE).read_text(encoding="utf-8"))
    assert packaged == spec_json_schema(), "run: python scripts/export_schema.py"


def test_repo_schema_is_current() -> None:
    root_copy = _PKG.parent.parent / SCHEMA_FILE
    if not root_copy.is_file():  # an installed wheel has no repo root
        return
    assert json.loads(root_copy.read_text(encoding="utf-8")) == spec_json_schema(), (
        "run: python scripts/export_schema.py"
    )
