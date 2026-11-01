"""Point 5: OS portability, verified rather than assumed. Static checks over the source (no I/O without an
explicit encoding, no separator assumptions), consoles that fall back to ASCII, line endings, BOMs, case,
and interpreter resolution for discover."""

from __future__ import annotations

import ast
import io
import os
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from agenticreq.capture.discover import DiscoverError, resolve_python
from agenticreq.engine import run, validate_spec
from agenticreq.loader import load_spec, load_spec_text
from agenticreq.observe import build_observed
from agenticreq.report import render_console
from agenticreq.report.term import AsciiFallback, make_console

SRC = Path(__file__).resolve().parents[1] / "src" / "agenticreq"
FIN = Path(__file__).parent / "fixtures" / "specs" / "real_world" / "fin_qa_agent.yml"
GOOD = Path(__file__).parent / "fixtures" / "specs" / "good.yml"


def _calls() -> list[tuple[Path, ast.Call]]:
    out = []
    for p in sorted(SRC.rglob("*.py")):
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Call):
                out.append((p, n))
    return out


def _name(n: ast.Call) -> str:
    return n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")


def test_no_text_io_without_an_explicit_encoding() -> None:
    bad = []
    for p, n in _calls():
        kws = {k.arg for k in n.keywords}
        name = _name(n)
        mode = next(
            (a.value for a in n.args[1:2] if isinstance(a, ast.Constant)),
            next((k.value.value for k in n.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)), "r"),
        )
        if name in ("read_text", "write_text") and "encoding" not in kws:
            bad.append(f"{p.name}:{n.lineno} {name}")
        elif name == "open" and isinstance(n.func, ast.Name) and "b" not in str(mode) and "encoding" not in kws:
            bad.append(f"{p.name}:{n.lineno} open")
        elif (
            name in ("run", "Popen", "check_output") and kws & {"text", "universal_newlines"} and "encoding" not in kws
        ):
            bad.append(f"{p.name}:{n.lineno} subprocess text mode")
    assert not bad, bad


def test_no_path_separator_assumptions() -> None:
    bad = []
    for p in sorted(SRC.rglob("*.py")):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(
                r"os\.sep in|in os\.sep|['\"]/tmp/|\.split\(['\"]\\\\\\\\['\"]\)|str\([^)]*\) \+ ['\"][/\\\\]", code
            ):
                bad.append(f"{p.name}:{i}: {line.strip()}")
    assert not bad, bad


# ------------------------------------------------------------------ consoles
@pytest.mark.parametrize("encoding", ["cp1252", "ascii", "latin-1"])
def test_reports_render_on_a_non_utf8_terminal(tmp_path: Path, encoding: str) -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding=encoding)  # strict: an unencodable character would raise
    con = make_console(file=stream, width=100)
    render_console(validate_spec(FIN), console=con)
    (tmp_path / "a.py").write_text("x = 'gpt-4o'\n", encoding="utf-8")
    render_console(run(mode="audit", observed=build_observed(tmp_path)), console=con)
    stream.flush()
    text = raw.getvalue().decode(encoding)
    assert "SPEC INVALID" in text and "audit without a spec" in text
    assert "+--" in text or "|" in text  # ASCII borders, not box drawing


def test_fallback_translates_instead_of_failing() -> None:
    raw = io.BytesIO()
    w = AsciiFallback(io.TextIOWrapper(raw, encoding="ascii"))
    w.write("a → b — c … ✓ 名前")
    w.flush()
    assert raw.getvalue().decode("ascii").startswith("a -> b - c ... ok ")


def test_no_colour_codes_when_not_a_terminal() -> None:
    buf = io.StringIO()
    render_console(validate_spec(FIN), console=make_console(file=buf))
    assert "\x1b[" not in buf.getvalue()


# ------------------------------------------------------------------ files: line endings, BOM, unicode, case
def test_crlf_and_bom_spec_load_with_the_same_lines(tmp_path: Path) -> None:
    lf = load_spec(GOOD)
    p = tmp_path / "crlf.yml"
    p.write_bytes(b"\xef\xbb\xbf" + GOOD.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    crlf = load_spec(p)
    assert crlf.status == lf.status == "valid"
    assert crlf.spec == lf.spec
    assert crlf.ref("prohibited", 0).line == lf.ref("prohibited", 0).line  # type: ignore[union-attr]


def test_unicode_spec_values(tmp_path: Path) -> None:
    p = tmp_path / "spec.yml"
    p.write_text(
        GOOD.read_text(encoding="utf-8").replace("agent_id: procurement-po-agent", "agent_id: ایجنٹ-café-代理"),
        encoding="utf-8",
    )
    assert load_spec(p).spec.agent_id == "ایجنٹ-café-代理"  # type: ignore[union-attr]


def test_case_insensitive_relative_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from agenticreq import pathutil

    monkeypatch.setattr(pathutil, "CASE_INSENSITIVE", True)
    assert pathutil.relative(Path("/Repo/Agent/tools.py"), Path("/repo/agent")) == Path("tools.py")
    assert pathutil.path_key(Path("/A/b.yml")) == pathutil.path_key(Path("/a/B.yml"))
    monkeypatch.setattr(pathutil, "CASE_INSENSITIVE", False)
    assert pathutil.relative(Path("/Repo/Agent/tools.py"), Path("/repo/agent")) is None


def test_windows_and_posix_paths_parse_the_same() -> None:
    # spec refs and extends targets are joined with pathlib only, so both spellings name one location
    assert PureWindowsPath("specs\\base.yml").parts == PurePosixPath("specs/base.yml").parts


def test_extends_with_windows_separators(tmp_path: Path) -> None:
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "b.yml").write_text("risk_level: low\n", encoding="utf-8")
    sep = "\\\\" if os.name == "nt" else "/"
    loaded = load_spec_text(f"extends: base{sep}b.yml\nagent_id: a\n", str(tmp_path / "a.yml"))
    assert loaded.spec is not None and loaded.spec.risk_level == "low"


# ------------------------------------------------------------------ discover: the interpreter
def test_resolve_python_absolute_relative_and_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = Path(sys.executable)
    assert resolve_python(None) == sys.executable
    assert resolve_python(str(exe)) == str(exe)
    monkeypatch.chdir(exe.parent)
    assert Path(resolve_python("." + "/" + exe.name)).samefile(exe)  # forward slashes work on Windows too
    monkeypatch.setenv("PATH", str(exe.parent) + os.pathsep + os.environ.get("PATH", ""))
    assert Path(resolve_python(exe.stem if os.name == "nt" else exe.name)).samefile(exe)


def test_resolve_python_missing_says_which_layout(tmp_path: Path) -> None:
    with pytest.raises(DiscoverError) as e:
        resolve_python(str(tmp_path / ".venv" / "bin" / "python"))
    msg = str(e.value)
    assert "bin" in msg and "Scripts" in msg and "python.exe" in msg
    with pytest.raises(DiscoverError, match="not on PATH"):
        resolve_python("python-that-does-not-exist-9")
