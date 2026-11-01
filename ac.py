"""Cross-platform launcher for agenticreq (Windows / macOS / Linux). Standard library only.

    python ac.py setup                 # create this OS's venv and install agenticreq into it
    python ac.py <agenticreq args>      # run agenticreq from that venv (sets it up on first use)

Each OS gets its own venv (.venv-windows / .venv-macos / .venv-linux), so a folder copied
between machines never runs a venv built for another OS. No activation needed.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OS_NAME = {"Windows": "windows", "Darwin": "macos"}.get(platform.system(), "linux")
VENV = ROOT / f".venv-{OS_NAME}"
BIN = VENV / ("Scripts" if os.name == "nt" else "bin")
EXE = ".exe" if os.name == "nt" else ""
PYTHON = BIN / f"python{EXE}"
AGENTICREQ = BIN / f"agenticreq{EXE}"


def setup(dev: bool = False) -> None:
    if sys.version_info < (3, 10):
        sys.exit(f"agenticreq needs Python >= 3.10 (this is {platform.python_version()})")
    if not PYTHON.exists():
        print(f"creating {VENV.name} ...")
        venv.EnvBuilder(with_pip=True, clear=True).create(VENV)
    target = ".[dev]" if dev else "."
    subprocess.run([str(PYTHON), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    subprocess.run([str(PYTHON), "-m", "pip", "install", "-q", "-e", target], cwd=ROOT, check=True)
    subprocess.run([str(AGENTICREQ), "version"], check=True)
    print(f"ready: python {Path(__file__).name} <command>   (venv: {VENV})")


def main(argv: list[str]) -> int:
    if argv[:1] == ["setup"]:
        setup(dev="--dev" in argv)
        return 0
    if not AGENTICREQ.exists():
        setup()
    return subprocess.run([str(AGENTICREQ), *argv]).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
