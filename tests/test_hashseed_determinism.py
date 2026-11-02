"""The report must not depend on PYTHONHASHSEED (set iteration order must never pick a path or a name)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def _audit(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed, "NO_COLOR": "1"}
    r = subprocess.run(
        [sys.executable, "-m", "agenticreq.cli", "audit", str(FIXTURES), "-f", "json", "--fail-on", "never"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_audit_output_is_the_same_under_different_hash_seeds() -> None:
    outputs = {seed: _audit(seed) for seed in ("0", "2")}
    assert len(set(outputs.values())) == 1, "the audit report changed with PYTHONHASHSEED"
