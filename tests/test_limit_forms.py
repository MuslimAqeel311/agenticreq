"""Caps written in any common form are found (ast/limits.py); a tool without one stays a gap; a cap whose value is
read at run time goes to "Check with a mock run". Each tool is named list_* / search_* so its name alone would make
it a candidate for "can return unbounded output"."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenticreq.ast.extractor import extract
from agenticreq.engine import run
from agenticreq.models.verdict import results
from agenticreq.observe import build_observed

HEAD = (
    "import itertools, os\nfrom langchain_core.tools import tool\nfrom pydantic import Field\n"
    "from typing import Annotated\n"
)

#: name -> (body, the number the report must give)
BOUNDED = {
    "list_slice": ("def list_slice(q: str) -> str:\n    rows = db(q)\n    return str(rows[:200])\n", "200"),
    "list_tail": ("def list_tail(q: str) -> str:\n    items = db(q)\n    return str(items[-50:])\n", "50"),
    "list_text": ("def list_text(q: str) -> str:\n    text = db(q)\n    return text[:5000]\n", "5000"),
    "list_islice": ("def list_islice(q: str) -> str:\n    return str(list(itertools.islice(db(q), 30)))\n", "30"),
    "list_counter": (
        "def list_counter(q: str) -> str:\n    out = []\n    for r in db(q):\n        out.append(r)\n"
        "        if len(out) >= 200:\n            break\n    return str(out)\n",
        "200",
    ),
    "list_enumerate": (
        "def list_enumerate(q: str) -> str:\n    out = []\n    for i, r in enumerate(db(q)):\n"
        "        if i >= 40:\n            break\n        out.append(r)\n    return str(out)\n",
        "40",
    ),
    "search_min": (
        "def search_min(q: str, max_results: int = 5) -> str:\n"
        "    n = max(1, min(int(max_results), 10))\n    return str(api(q, max_results=n))\n",
        "10",
    ),
    "list_raise": (
        "def list_raise(q: str) -> str:\n    rows = db(q)\n    if len(rows) > 500:\n"
        "        raise ValueError('too many')\n    return str(rows)\n",
        "500",
    ),
    "list_sql": (
        "def list_sql(q: str) -> str:\n    return str(cur.execute('SELECT * FROM t WHERE a = ? LIMIT 25', (q,)))\n",
        "25",
    ),
    "list_fsql": (
        "def list_fsql(q: str) -> str:\n    n = 15\n    return str(cur.execute(f'SELECT * FROM t LIMIT {n}'))\n",
        "15",
    ),
    "list_orm": (
        "def list_orm(q: str) -> str:\n    return str(session.query(T).filter_by(a=q).limit(20).all())\n",
        "20",
    ),
    "list_head": ("def list_head(q: str) -> str:\n    return df(q).head(12).to_string()\n", "12"),
    "list_nlargest": ("def list_nlargest(q: str) -> str:\n    return df(q).nlargest(7, 'x').to_string()\n", "7"),
    "search_api": ("def search_api(q: str) -> str:\n    return str(client.search(q, top_k=8))\n", "8"),
    "search_page": ("def search_page(q: str) -> str:\n    return str(client.search(q, page_size=25))\n", "25"),
    "list_const": (
        "MAX_ROWS = 200\n\ndef list_const(q: str) -> str:\n    return str(db(q)[:MAX_ROWS])\n",
        "200 = MAX_ROWS",
    ),
    "list_imported": (
        "from limits_mod import MAX_ITEMS\n\ndef list_imported(q: str) -> str:\n    return str(db(q)[:MAX_ITEMS])\n",
        "60 = MAX_ITEMS",
    ),
    "list_helper": (
        "def _cut(text: str, limit: int) -> str:\n    return text[:limit]\n\n"
        "def list_helper(q: str) -> str:\n    return _cut(db(q), 3000)\n",
        "3000",
    ),
    "list_class": (
        "class Store:\n    MAX = 75\n\n    def list_class(self, q: str) -> str:\n"
        "        return str(db(q)[:self.MAX])\n",
        "75",
    ),
}


def _project(tmp: Path, body: str) -> Path:
    (tmp / "limits_mod.py").write_text("MAX_ITEMS = 60\n", encoding="utf-8")
    name = [d.split("(")[0] for d in body.split("def ")[1:] if not d.startswith("_")][0]  # the public one is the tool
    decorated = body.replace(f"def {name}(", f"@tool\ndef {name}(")
    (tmp / "agent.py").write_text(HEAD + decorated, encoding="utf-8")
    return tmp


@pytest.mark.parametrize("name", sorted(BOUNDED))
def test_each_form_is_a_cap(name: str, tmp_path: Path) -> None:
    body, expect = BOUNDED[name]
    if name == "list_class":  # a method: find it without a framework, as a tool registered elsewhere would be
        from agenticreq.ast.limits import find_limits
        from agenticreq.ast.symbols import FileCtx  # noqa: F401
        import ast as _ast

        tree = _ast.parse(body)
        fn = tree.body[0].body[1]  # type: ignore[attr-defined]

        class _FC:
            classes = {"Store": tree.body[0]}

        lims = find_limits(None, _FC(), fn)  # type: ignore[arg-type]
        assert lims and lims[0].value == 75 and lims[0].source == "Store.MAX"
        return
    obs = extract(_project(tmp_path, body))
    t = obs.declared_tools[name]
    assert t.limits, (name, t.features)
    assert any(f"capped at {expect}" in x for x in t.limits), t.limits
    assert "bounded_output" in t.features
    r = run(mode="audit", observed=build_observed(tmp_path))
    assert not [f for f in r.findings if f.rule_id == "AS-213" and name in f.title]  # never a gap
    line = next((ln for ln in results(r) if ln.param == "output_size_limits"), None)
    assert line is None or (line.status == "PASS" and "capped" in line.reason)  # None: its limit param already said so


UNBOUNDED = {
    "list_all": "def list_all(q: str) -> str:\n    return str(cur.execute('SELECT * FROM t').fetchall())\n",
    "search_everything": "def search_everything(query: str) -> str:\n    return str(client.search(query))\n",
}


@pytest.mark.parametrize("name", sorted(UNBOUNDED))
def test_a_tool_with_no_cap_is_still_a_gap(name: str, tmp_path: Path) -> None:
    _project(tmp_path, UNBOUNDED[name])
    r = run(mode="audit", observed=build_observed(tmp_path))
    gaps = [f for f in r.findings if f.rule_id == "AS-213" and name in f.title]
    assert gaps and gaps[0].title == f"{name} can return unbounded output"
    assert next(ln for ln in results(r) if ln.param == "output_size_limits").status == "GAP"


@pytest.mark.parametrize(
    "body,source",
    [
        (
            "def list_env(q: str) -> str:\n    n = int(os.getenv('MAX_ROWS', '50'))\n    return str(db(q)[:n])\n",
            "env MAX_ROWS",
        ),
        ("def list_env2(q: str) -> str:\n    return str(db(q)[:int(os.environ['ROW_CAP'])])\n", "env ROW_CAP"),
        ("def list_cfg(q: str) -> str:\n    return str(db(q)[:settings.max_rows])\n", "settings.max_rows"),
    ],
)
def test_a_cap_read_at_run_time_goes_to_a_mock_run(body: str, source: str, tmp_path: Path) -> None:
    _project(tmp_path, body)
    t = next(iter(extract(tmp_path).declared_tools.values()))
    assert t.limit_runtime == source and "bounded_output" not in t.features
    r = run(mode="audit", observed=build_observed(tmp_path))
    line = next(ln for ln in results(r) if ln.param == "output_size_limits")
    assert line.status == "CHECK" and f"limit set from {source}" in line.reason
    assert not [f for f in r.findings if f.rule_id == "AS-213" and f.title.endswith("can return unbounded output")]


def test_a_slice_inside_a_test_is_not_a_cap(tmp_path: Path) -> None:
    _project(
        tmp_path,
        "def list_peek(q: str) -> str:\n    body = db(q)\n    if '<html' in body[:500]:\n"
        "        body = strip(body)\n    return body\n",
    )
    assert extract(tmp_path).declared_tools["list_peek"].limits == []


def test_an_argument_cap_names_the_argument(tmp_path: Path) -> None:
    _project(tmp_path, "def list_news(q: str, limit: int = 5) -> str:\n    return str(db(q)[:limit])\n")
    (lim,) = extract(tmp_path).declared_tools["list_news"].limits
    assert lim.startswith("capped by the limit argument (default 5)")


@pytest.mark.parametrize(
    "param",
    [
        "n: Annotated[int, Field(le=50)]",
        "q: Annotated[str, Field(max_length=200)]",
        "items: Annotated[list[str], Field(max_length=10)]",
    ],
)
def test_a_constrained_parameter_is_a_cap(param: str, tmp_path: Path) -> None:
    _project(tmp_path, f"def search_constrained({param}) -> str:\n    return str(client.search(q='x'))\n")
    r = run(mode="audit", observed=build_observed(tmp_path))
    assert not [f for f in r.findings if f.rule_id == "AS-213" and "search_constrained" in f.title]
