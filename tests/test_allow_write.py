"""allow_write: in mocks.yml: the paths a mock run may write to, relative to the project folder.

Paths are compared after resolving / and \\, `..` and symlinks, and case-insensitively on Windows; an allowed folder
includes everything under it; a file outside it is blocked even when reached through ../ or a symlink. Without
allow_write: only .agenticreq/ and the temp folder may be written. Writes inside allowed paths are listed in the
summary and are not a change of the project's files."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agenticreq.capture.guard import ProcessGuard, norm_path, under
from agenticreq.cli import app
from agenticreq.mockgen import write_folders

runner = CliRunner()


def _guard(tmp: Path, *roots: str) -> ProcessGuard:
    return ProcessGuard([str(tmp / r) for r in roots], cache_dirs=False, project=str(tmp))


def test_an_allowed_file_and_an_allowed_folder(tmp_path: Path) -> None:
    g = _guard(tmp_path, "workspace", "notes.md")
    assert g._under_roots(str(tmp_path / "workspace" / "final_answer.md"))
    assert g._under_roots(str(tmp_path / "workspace" / "deep" / "x.txt"))  # everything under the folder
    assert g._under_roots(str(tmp_path / "notes.md"))
    assert not g._under_roots(str(tmp_path / "report.md"))
    assert not g._under_roots(str(tmp_path / "workspace2" / "x.txt"))  # a sibling with the same prefix is not inside


def test_backslash_and_forward_slash_are_the_same(tmp_path: Path) -> None:
    roots = [norm_path(tmp_path / "workspace")]
    assert under(str(tmp_path) + "/workspace/a.md", roots)
    if os.name == "nt":
        assert under(str(tmp_path) + "\\workspace\\a.md", roots)


@pytest.mark.skipif(os.name != "nt", reason="Windows paths are case-insensitive")
def test_mixed_case_on_windows(tmp_path: Path) -> None:
    assert under(str(tmp_path / "WorkSpace" / "A.md"), [norm_path(tmp_path / "workspace")])


def test_a_dotdot_escape_is_blocked(tmp_path: Path) -> None:
    g = _guard(tmp_path, "workspace")
    assert not g._under_roots(str(tmp_path / "workspace" / ".." / "report.md"))  # lands in the project root
    assert not g._under_roots(str(tmp_path / "workspace" / "sub" / ".." / ".." / "other" / "x"))


def test_a_symlink_escape_is_blocked(tmp_path: Path) -> None:
    (tmp_path / "workspace").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "workspace" / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow creating symlinks")
    g = _guard(tmp_path, "workspace")
    assert not g._under_roots(str(link / "secret.txt"))  # resolves to outside/secret.txt


def test_no_allow_write_means_only_agenticreq_and_temp(tmp_path: Path) -> None:
    import tempfile

    g = ProcessGuard([str(tmp_path / ".agenticreq")], cache_dirs=False, project=str(tmp_path))  # as the runner
    assert g._under_roots(str(tmp_path / ".agenticreq" / "captures.json"))
    assert g._under_roots(os.path.join(tempfile.gettempdir(), "x"))
    assert not g._under_roots(str(tmp_path / "anything.txt"))
    assert not g._under_roots(os.path.join(os.path.expanduser("~"), ".cache", "lib", "x"))  # no cache folders


def _mock_run(tmp: Path, body: str, monkeypatch: pytest.MonkeyPatch, allow: str | None) -> tuple[int, str]:
    (tmp / "agent.py").write_text(body, encoding="utf-8")
    lines = ["entry: agent.py:main", "tools: {}"] + ([f"allow_write: {allow}"] if allow is not None else [])
    (tmp / "mocks.yml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp)
    r = runner.invoke(app, ["mock-run", "--quiet"], terminal_width=200)
    return r.exit_code, " ".join(r.output.split())


WRITE_ANSWER = (
    "import pathlib\ndef main():\n    p = pathlib.Path('workspace')\n    p.mkdir(exist_ok=True)\n"
    "    (p / 'final_answer.md').write_text('x' * 2048, encoding='utf-8')\n"
)


@pytest.mark.parametrize("allow", ["[workspace/]", "[workspace]", "['workspace\\\\']", "[./workspace]"])
def test_zero_touch_with_allowed_writes(allow: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, out = _mock_run(tmp_path, WRITE_ANSWER, monkeypatch, allow)
    assert "wrote workspace/final_answer.md, 2 KB" in out, out
    assert "changed the agent's files" not in out


def test_without_allow_write_a_project_write_is_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "workspace").mkdir()  # the folder exists; the agent tries to write a file into it
    body = (
        "import pathlib\ndef main():\n    try:\n"
        "        (pathlib.Path('workspace') / 'final_answer.md').write_text('x')\n"
        "    except PermissionError:\n        pass\n"
    )
    _, out = _mock_run(tmp_path, body, monkeypatch, None)
    assert "blocked write to workspace/final_answer.md" in out
    assert not (tmp_path / "workspace" / "final_answer.md").exists()


def test_mocks_prefills_folders_the_code_writes_to(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        "import os\nfrom pathlib import Path\nOUT = 'reports'\n"
        "def save(name, text):\n    (Path(OUT) / name).write_text(text)\n"
        "def log(line):\n    with open('logs/run.log', 'a') as fh:\n        fh.write(line)\n"
        "def answer(t):\n    d = os.getenv('ANSWER_DIR', 'workspace')\n    Path(d, 'a.md').write_text(t)\n",
        encoding="utf-8",
    )
    assert set(write_folders(tmp_path)) == {"reports", "logs", "workspace"}
    from agenticreq.ast.extractor import extract
    from agenticreq.mockgen import render_mocks

    text = render_mocks(tmp_path, extract(tmp_path))
    line = next(ln for ln in text.splitlines() if ln.startswith("allow_write:"))
    assert "logs/" in line and "reports/" in line and "workspace/" in line


def test_mocks_writes_an_empty_allow_write_with_a_comment(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    from agenticreq.ast.extractor import extract
    from agenticreq.mockgen import render_mocks

    line = next(ln for ln in render_mocks(tmp_path, extract(tmp_path)).splitlines() if ln.startswith("allow_write:"))
    assert line.startswith("allow_write: []  # ")


assert sys.version_info >= (3, 10)


def test_parent_folders_of_an_allowed_folder_can_be_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """allow_write: [data/mock_run] with no data/ folder yet: mkdir(parents=True) must create data/ first."""
    g = _guard(tmp_path, "data/mock_run")
    assert g._leads_to_allowed(str(tmp_path / "data"))
    assert not g._leads_to_allowed(str(tmp_path / "other"))
    assert not g._under_roots(str(tmp_path / "data" / "file.txt"))  # only the folder may be made, not files in it
    body = (
        "import pathlib\ndef main():\n    d = pathlib.Path('data/mock_run/blobs')\n"
        "    d.mkdir(parents=True, exist_ok=True)\n    (d / 'x.bin').write_bytes(b'1')\n"
    )
    _, out = _mock_run(tmp_path, body, monkeypatch, "[data/mock_run]")
    assert "blocked" not in out and "wrote data/mock_run/blobs/x.bin" in out, out
