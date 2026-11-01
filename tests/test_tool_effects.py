"""Each tool is READ, WRITE or UNCLEAR by what its body does (ast/effects.py), never by its name when the body can be
read. UNCLEAR goes to "Check with a mock run", never to a gap. tool_effects: in spec.yml decides an unclear tool, and
a body that writes against a declared read is a gap with the line that writes."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenticreq.ast.classify import is_write_tool
from agenticreq.ast.extractor import extract
from agenticreq.engine import run
from agenticreq.loader import load_spec_text
from agenticreq.models.verdict import results
from agenticreq.observe import build_observed

HEAD = (
    "import os, shutil, subprocess, smtplib, sqlite3\nimport requests, httpx, boto3\nfrom pathlib import Path\n"
    "from langchain_core.tools import tool\n"
)


def _tool(tmp: Path, name: str, body: str, params: str = "x: str") -> object:
    lines = "\n".join("    " + ln for ln in body.splitlines())
    (tmp / "agent.py").write_text(
        f"{HEAD}\n@tool\ndef {name}({params}) -> str:\n    '''Do it.'''\n{lines}\n", encoding="utf-8"
    )
    return extract(tmp).declared_tools[name]


# every name is a neutral verb, so the result can only come from the body
WRITES = {
    "open_w": "with open(x, 'w') as fh:\n    fh.write('a')\nreturn 'ok'",
    "open_a": "open(x, 'a').write('a')\nreturn 'ok'",
    "open_x": "open(x, mode='x')\nreturn 'ok'",
    "open_plus": "open(x, 'r+')\nreturn 'ok'",
    "write_text": "Path(x).write_text('a')\nreturn 'ok'",
    "write_bytes": "Path(x).write_bytes(b'a')\nreturn 'ok'",
    "os_remove": "os.remove(x)\nreturn 'ok'",
    "os_rename": "os.rename(x, x + '.bak')\nreturn 'ok'",
    "os_makedirs": "os.makedirs(x)\nreturn 'ok'",
    "shutil_rmtree": "shutil.rmtree(x)\nreturn 'ok'",
    "shutil_copy": "shutil.copy(x, '/tmp/y')\nreturn 'ok'",
    "subprocess": "subprocess.run(['ls', x])\nreturn 'ok'",
    "os_system": "os.system(x)\nreturn 'ok'",
    "http_post": "requests.post('https://api.example', json={'x': x})\nreturn 'ok'",
    "http_put": "httpx.put('https://api.example', json={'x': x})\nreturn 'ok'",
    "http_delete": "requests.delete('https://api.example/' + x)\nreturn 'ok'",
    "sql_insert": "sqlite3.connect('db').execute('INSERT INTO t VALUES (?)', (x,))\nreturn 'ok'",
    "sql_update": "cur = sqlite3.connect('db').cursor()\ncur.execute(\"UPDATE t SET a = 1\")\nreturn 'ok'",
    "orm_commit": "session.add(Row(x))\nsession.commit()\nreturn 'ok'",
    "orm_save": "obj = Row(x)\nobj.save()\nreturn 'ok'",
    "email": "smtplib.SMTP('mail').send_message(msg)\nreturn 'ok'",
    "cloud_put": "boto3.client('s3').put_object(Bucket='b', Key=x, Body=b'')\nreturn 'ok'",
    "store_async": "store.aput('k', x, {})\nreturn 'ok'",
    "state": "state.seat = x\nreturn 'ok'",
}
READS = {
    "open_r": "return open(x).read()",
    "open_rb": "return str(open(x, 'rb').read())",
    "read_text": "return Path(x).read_text()",
    "listdir": "return str(os.listdir(x))",
    "http_get": "return requests.get('https://api.example', params={'q': x}).text",
    "sql_select": "return str(sqlite3.connect('db').execute('SELECT * FROM t WHERE a = ?', (x,)).fetchall())",
}
PURE = {
    "math": "import math\nreturn str(math.sqrt(len(x)))",
    "strings": "words = [w.strip('.,').lower() for w in x.split()]\nreturn ', '.join(sorted(set(words)))",
    "parsing": "import json\nreturn json.dumps({'n': int(x)})",
    "stub": "return 'PO-1'",
}


@pytest.mark.parametrize("case", sorted(WRITES))
def test_write_evidence(case: str, tmp_path: Path) -> None:
    params = "x: str, state: object" if case == "state" else "x: str"
    t = _tool(tmp_path, f"handle_{case}", WRITES[case], params)
    assert t.effect == "write", (case, t.effect, t.effect_reason)  # type: ignore[attr-defined]
    assert t.effect_line and is_write_tool(t)  # type: ignore[attr-defined, arg-type]


@pytest.mark.parametrize("case", sorted(READS))
def test_read_evidence(case: str, tmp_path: Path) -> None:
    t = _tool(tmp_path, f"handle_{case}", READS[case])
    assert t.effect == "read" and t.effect_reason.startswith("reads only"), (case, t.effect_reason)  # type: ignore[attr-defined]
    assert not is_write_tool(t)  # type: ignore[arg-type]


@pytest.mark.parametrize("case", sorted(PURE))
def test_a_pure_function_is_read(case: str, tmp_path: Path) -> None:
    t = _tool(tmp_path, f"handle_{case}", PURE[case])
    assert (t.effect, t.effect_reason) == ("read", "no side effects found")  # type: ignore[attr-defined]


def test_the_name_never_overrides_the_body(tmp_path: Path) -> None:
    assert _tool(tmp_path, "delete_record", "return x.upper()").effect == "read"  # type: ignore[attr-defined]
    assert _tool(tmp_path, "get_report", "Path(x).write_text('a')\nreturn 'ok'").effect == "write"  # type: ignore[attr-defined]


def test_mkdir_exist_ok_only_makes_sure_a_folder_exists(tmp_path: Path) -> None:
    t = _tool(tmp_path, "list_things", "Path(x).mkdir(parents=True, exist_ok=True)\nreturn str(os.listdir(x))")
    assert t.effect == "read"  # type: ignore[attr-defined]


def test_project_helpers_are_followed(tmp_path: Path) -> None:
    (tmp_path / "helpers.py").write_text(
        "def _save(p, t):\n    with open(p, 'w') as fh:\n        fh.write(t)\n", encoding="utf-8"
    )
    (tmp_path / "agent.py").write_text(
        "from langchain_core.tools import tool\nfrom helpers import _save\n"
        "@tool\ndef handle(x: str) -> str:\n    '''Do it.'''\n    _save(x, 'a')\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    t = extract(tmp_path).declared_tools["handle"]
    assert t.effect == "write" and "(in _save)" in t.effect_reason


def test_an_unknown_library_call_is_unclear_and_never_a_gap(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        "import stripe\nfrom langchain_core.tools import tool\n"
        "@tool\ndef send_payment(amount: float) -> str:\n    '''Pay.'''\n"
        "    stripe.Charge.create(amount=amount)\n    return 'ok'\n",
        encoding="utf-8",
    )
    t = extract(tmp_path).declared_tools["send_payment"]
    assert t.effect == "unclear" and "stripe.Charge.create" in t.effect_reason
    r = run(mode="audit", observed=build_observed(tmp_path))
    lines = [ln for ln in results(r) if ln.param == "write_safety"]
    assert lines and all(ln.status == "CHECK" for ln in lines)
    assert "stripe.Charge.create" in lines[0].reason and "agenticreq mock-run" in lines[0].reason
    assert not [f for f in r.findings if f.rule_id == "AS-212" and f.title.startswith("Write tool")]


def test_a_placeholder_body_is_unclear(tmp_path: Path) -> None:
    t = _tool(tmp_path, "send_refund", "raise RuntimeError('not here')")
    assert t.effect == "unclear" and "placeholder" in t.effect_reason  # type: ignore[attr-defined]


SPEC = """agent_id: a
risk_level: high
goal: {statement: s, done_criteria: [d]}
out_of_scope: []
prohibited: []
approval_required: []
limits: []
escalation: []
untrusted_inputs: []
owner: o@example.com
technical_owner: t@example.com
"""


def test_declared_read_but_body_writes_is_a_gap_with_the_line(tmp_path: Path) -> None:
    _tool(tmp_path, "read_file", "Path(x).write_text('a')\nreturn 'ok'")
    r = run(
        mode="check",
        loaded=load_spec_text(SPEC + "tool_effects: {read_file: read}\n"),
        observed=build_observed(tmp_path),
    )
    (f,) = [f for f in r.findings if f.rule_id == "AS-232"]
    assert f.title == "read_file is declared read, but it writes" and "agent.py:" in (f.reason or "")
    assert next(ln for ln in results(r) if ln.param == "tool_effects").status == "GAP"


def test_a_declaration_decides_an_unclear_tool(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(
        "import yfinance as yf\nfrom langchain_core.tools import tool\n"
        "@tool\ndef get_quote(t: str) -> str:\n    '''Quote.'''\n    return str(yf.Ticker(t).info)\n",
        encoding="utf-8",
    )
    r = run(
        mode="check",
        loaded=load_spec_text(SPEC + "tool_effects: {get_quote: read}\n"),
        observed=build_observed(tmp_path),
    )
    assert not [f for f in r.findings if f.rule_id in ("AS-212", "AS-232") and "get_quote" in f.title]


def test_the_classification_reaches_every_write_rule(tmp_path: Path) -> None:
    """approval (AS-212), dry run (AS-227), trust boundary (AS-401) all use the body's answer."""
    _tool(tmp_path, "process_item", "requests.post('https://api.example', json={'x': x})\nreturn 'ok'")
    r = run(mode="check", loaded=load_spec_text(SPEC), observed=build_observed(tmp_path))
    ids = {f.rule_id for f in r.findings if "process_item" in f.title}
    assert {"AS-212", "AS-227"} <= ids
