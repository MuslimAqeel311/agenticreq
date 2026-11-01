"""YAML → AgentSpec with a line number for every value.

* duplicate mapping keys are detected (PyYAML silently keeps the last one)
* ``TODO`` placeholders are recorded and treated as missing
* ``extends`` (file path or ``profile:<name>``) is deep-merged, parents first;
  safety lists (prohibited, approval_required, ...) are unioned, never replaced
* short forms are normalised (``goal: "..."``, ``prohibited: [tool]``, dict-form limits)
* prose where structure is required is kept as a note; the field stays undeclared (``loader/issues.py``)
* shape errors become ``SpecIssue``s (collapsed, each with the expected shape and an example): an invalid
  spec stops `check` / `validate` before any rule runs
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agenticreq.loader.issues import (
    errors_to_issues,
    extract_prose,
    is_placeholder,
    placeholder_issue,
    precheck,
    prose_issue,
)
from agenticreq.models.confidence import Confidence
from agenticreq.models.finding import Finding, SourceRef, SpecIssue
from agenticreq.models.spec import AgentSpec
from agenticreq.not_needed import SHAPE_HINT
from agenticreq.pathutil import path_key
from agenticreq.rules.registry import make_finding
from agenticreq.shapes import example_yaml

PathKey = tuple[str | int, ...]

PROFILES_DIR = Path(__file__).resolve().parent.parent / "defaults" / "profiles"

#: keys whose lists are unioned across `extends` (a child must not silently drop a parent's safety rule)
UNION_KEYS = frozenset({"prohibited", "approval_required", "out_of_scope", "untrusted_inputs", "confirmed"})


def is_todo(v: Any) -> bool:
    return isinstance(v, str) and v.strip().upper().startswith("TODO")


@dataclass
class LoadedSpec:
    path: Path
    raw: dict[str, Any]
    refs: dict[PathKey, SourceRef]
    spec: AgentSpec | None
    todos: list[PathKey] = field(default_factory=list)
    load_findings: list[Finding] = field(default_factory=list)
    user_tier2: dict[str, Any] = field(default_factory=dict)
    extends_chain: list[str] = field(default_factory=list)
    #: problems with the file itself: ``invalid`` ones stop every rule; the others leave a field undeclared
    issues: list[SpecIssue] = field(default_factory=list)
    #: prose kept from structured fields (``role_boundary: "read-only ..."``), never treated as satisfied
    notes: dict[str, str] = field(default_factory=dict)
    #: parameters written as ``not_needed`` (not_needed.py), with where; removed from ``raw`` and the model
    not_needed: dict[str, SourceRef] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if any(i.kind == "invalid" for i in self.issues):
            return "invalid"
        if self.issues or self.todos:
            return "incomplete"
        return "valid"

    @property
    def invalid(self) -> list[SpecIssue]:
        return [i for i in self.issues if i.kind == "invalid"]

    def ref(self, *path: str | int) -> SourceRef | None:
        """SourceRef for a path, falling back to the nearest ancestor."""
        p: PathKey = tuple(path)
        while p:
            if p in self.refs:
                return self.refs[p]
            p = p[:-1]
        return SourceRef(file=str(self.path), line=1)


# ---------------------------------------------------------------- YAML with lines
class _NodeWalker:
    def __init__(self, file: str) -> None:
        self.file = file
        self.refs: dict[PathKey, SourceRef] = {}
        self.duplicates: list[tuple[PathKey, SourceRef]] = []
        self._loader = yaml.SafeLoader("")

    def walk(self, node: yaml.Node, path: PathKey) -> Any:
        self.refs[path] = SourceRef(file=self.file, line=node.start_mark.line + 1)
        if isinstance(node, yaml.MappingNode):
            out: dict[str, Any] = {}
            for knode, vnode in node.value:
                key = self._loader.construct_object(knode, deep=True)
                if not isinstance(key, (str, int, float, bool)):
                    key = str(key)
                skey = str(key)
                kref = SourceRef(file=self.file, line=knode.start_mark.line + 1)
                if skey in out:
                    self.duplicates.append((path + (skey,), kref))
                out[skey] = self.walk(vnode, path + (skey,))
                self.refs[path + (skey,)] = kref
            return out
        if isinstance(node, yaml.SequenceNode):
            return [self.walk(n, path + (i,)) for i, n in enumerate(node.value)]
        return self._loader.construct_object(node, deep=True)


def _parse_yaml(
    text: str, file: str
) -> tuple[dict[str, Any], dict[PathKey, SourceRef], list[tuple[PathKey, SourceRef]]]:
    node = yaml.compose(text, Loader=yaml.SafeLoader)
    if node is None:
        return {}, {}, []
    w = _NodeWalker(file)
    data = w.walk(node, ())
    if not isinstance(data, dict):
        raise ValueError("spec root must be a mapping")
    return data, w.refs, w.duplicates


# ---------------------------------------------------------------- extends
def _merge_raw(
    parent: dict[str, Any],
    prefs: dict[PathKey, SourceRef],
    child: dict[str, Any],
    crefs: dict[PathKey, SourceRef],
) -> tuple[dict[str, Any], dict[PathKey, SourceRef]]:
    out = copy.deepcopy(parent)
    refs = dict(prefs)

    def rec(dst: dict[str, Any], src: dict[str, Any], path: PathKey, top: bool) -> None:
        for k, v in src.items():
            if top and k in UNION_KEYS and isinstance(v, list) and isinstance(dst.get(k), list):
                base = dst[k]
                seen = [repr(x) for x in base]
                for i, item in enumerate(v):
                    if repr(item) not in seen:
                        base.append(copy.deepcopy(item))
                        j = len(base) - 1
                        for rp, r in crefs.items():
                            if rp[: len(path) + 2] == path + (k, i):
                                refs[path + (k, j) + rp[len(path) + 2 :]] = r
                continue
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                rec(dst[k], v, path + (k,), False)
                if path + (k,) in crefs:
                    refs[path + (k,)] = crefs[path + (k,)]
                continue
            dst[k] = copy.deepcopy(v)
            for rp in [rp for rp in refs if rp[: len(path) + 1] == path + (k,)]:
                del refs[rp]
            for rp, r in crefs.items():
                if rp[: len(path) + 1] == path + (k,):
                    refs[rp] = r

    rec(out, child, (), True)
    return out, refs


def _resolve_extends_target(target: str, base_dir: Path) -> Path:
    if target.startswith("profile:"):
        return PROFILES_DIR / f"{target.split(':', 1)[1]}.yml"
    p = Path(target)
    return p if p.is_absolute() else (base_dir / p)


def _load_raw_with_extends(
    path: Path,
    text: str | None,
    stack: list[str],
    findings: list[Finding],
    chain: list[str],
) -> tuple[dict[str, Any], dict[PathKey, SourceRef], list[tuple[PathKey, SourceRef]]]:
    if text is None:
        text = path.read_text(encoding="utf-8-sig")  # a BOM (Windows editors) is not part of the spec
    raw, refs, dups = _parse_yaml(text, str(path))
    ext = raw.get("extends")
    targets = [ext] if isinstance(ext, str) else (ext if isinstance(ext, list) else [])
    merged: dict[str, Any] = {}
    mrefs: dict[PathKey, SourceRef] = {}
    for t in targets:
        if not isinstance(t, str):
            continue
        tp = _resolve_extends_target(t, path.parent)
        key = path_key(tp.resolve())  # case-insensitive on Windows and default macOS
        ref = refs.get(("extends",), SourceRef(file=str(path), line=1))
        if key in stack:
            findings.append(
                make_finding(
                    "AS-115",
                    confidence=Confidence.proven(),
                    spec_ref=ref,
                    detail=f"extends cycle: {' → '.join([*stack, key])}",
                    evidence=f"extends: {t}",
                )
            )
            continue
        if not tp.exists():
            findings.append(
                make_finding(
                    "AS-115",
                    confidence=Confidence.proven(),
                    spec_ref=ref,
                    detail=f"extends target {t!r} not found (looked at {tp})",
                    evidence=f"extends: {t}",
                )
            )
            continue
        chain.append(str(tp))
        praw, prefs, pdups = _load_raw_with_extends(tp, None, [*stack, key], findings, chain)
        dups.extend(pdups)
        merged, mrefs = _merge_raw(merged, mrefs, praw, prefs)
    if merged:
        child = {k: v for k, v in raw.items() if k != "extends"}
        raw2, refs2 = _merge_raw(merged, mrefs, child, refs)
        if "extends" in raw:
            raw2["extends"] = raw["extends"]
        return raw2, refs2, dups
    return raw, refs, dups


# ---------------------------------------------------------------- normalisation
def _strip_todos(
    v: Any, path: PathKey, todos: list[PathKey], placeholders: list[tuple[PathKey, str]] | None = None
) -> Any:
    if is_todo(v):
        todos.append(path)
        return None
    if placeholders is not None and is_placeholder(v) and path and path[0] != "_derived":
        placeholders.append((path, str(v)))  # ${OWNER}: not expanded, so the value is not there
        return None
    if isinstance(v, dict):
        return {k: _strip_todos(x, path + (k,), todos, placeholders) for k, x in v.items()}
    if isinstance(v, list):
        return [_strip_todos(x, path + (i,), todos, placeholders) for i, x in enumerate(v)]
    return v


def _normalise(raw: dict[str, Any], refs: dict[PathKey, SourceRef]) -> dict[str, Any]:
    d = copy.deepcopy(raw)

    g = d.get("goal")
    if isinstance(g, str):
        d["goal"] = {"statement": g}

    for key, field_name in (("prohibited", "tool"), ("approval_required", "tool"), ("untrusted_inputs", "source")):
        v = d.get(key)
        if isinstance(v, dict):  # {tool: {...}} form
            items = []
            for i, (name, body) in enumerate(v.items()):
                items.append({field_name: name, **(body if isinstance(body, dict) else {})})
                if (key, name) in refs:
                    refs[(key, i)] = refs[(key, name)]
            d[key] = items
        elif isinstance(v, list):
            d[key] = [{field_name: x} if isinstance(x, str) else x for x in v]

    lim = d.get("limits")
    if isinstance(lim, dict):
        items = []
        for tool, args in lim.items():
            if not isinstance(args, dict):
                items.append({"tool": tool, "arg": "?", "limit": args})
                continue
            for arg, spec in args.items():
                if (("limits", tool, arg)) in refs:
                    refs[("limits", len(items))] = refs[("limits", tool, arg)]
                items.append({"tool": tool, "arg": arg, "limit": spec})
        d["limits"] = items
    elif isinstance(lim, list):
        items = []
        for x in lim:
            if isinstance(x, dict) and "limit" not in x:
                x = {
                    "tool": x.get("tool"),
                    "arg": x.get("arg"),
                    "limit": {k: v for k, v in x.items() if k not in ("tool", "arg")},
                }
            items.append(x)
        d["limits"] = items

    dr = d.get("data_rules")
    if isinstance(dr, dict) and isinstance(dr.get("retention"), str):
        dr["retention"] = {"*": dr["retention"]}

    dd = d.get("declared_delegation")
    if isinstance(dd, list):
        out = []
        for e in dd:
            if isinstance(e, str) and "->" in e:
                a, b = (s.strip() for s in e.split("->", 1))
                out.append({"from": a, "to": b})
            else:
                out.append(e)
        d["declared_delegation"] = out

    # prose escalation is reported by AS-108 from raw; drop it for model construction
    esc = d.get("escalation")
    if isinstance(esc, str) or (isinstance(esc, list) and any(isinstance(e, str) for e in esc)):
        d["escalation"] = [e for e in esc if isinstance(e, dict)] if isinstance(esc, list) else None
    return d


def _build_model(
    data: dict[str, Any],
    loaded_refs: dict[PathKey, SourceRef],
    file: str,
    errors: list[tuple[dict[str, Any], SourceRef]],
    todos: frozenset[PathKey] = frozenset(),
) -> AgentSpec | None:
    """Validate; on errors, record them and retry without the offending field (so the rest still loads)."""
    current = dict(data)
    for _ in range(20):
        try:
            return AgentSpec.model_validate(current)
        except ValidationError as e:
            dropped = False
            for err in e.errors():
                loc = tuple(err["loc"])
                top = str(loc[0]) if loc else ""
                ref = None
                p: PathKey = tuple(loc)
                while p and ref is None:
                    ref = loaded_refs.get(p)
                    p = p[:-1]
                ref = ref or SourceRef(file=file, line=1)
                if not any(tuple(loc)[: len(t)] == t for t in todos if t):
                    # a TODO left in place is already AS-101; its type error would only repeat it
                    errors.append((dict(err), ref))
                if top in current:
                    if len(loc) > 1 and isinstance(current[top], dict) and err["type"] == "extra_forbidden":
                        sub = copy.deepcopy(current[top])
                        cur: Any = sub
                        for part in loc[1:-1]:
                            cur = cur[part] if isinstance(cur, dict) else cur[int(part)]
                        if isinstance(cur, dict):
                            cur.pop(str(loc[-1]), None)
                        current[top] = sub
                    else:
                        del current[top]
                    dropped = True
            if not dropped:
                return None
    return None


_TIMEOUT = re.compile(r"\d+\s*[smhd]")


def _timeout_issues(norm: dict[str, Any], refs: dict[PathKey, SourceRef], file: str) -> list[SpecIssue]:
    bad = [
        (i, a["timeout"])
        for i, a in enumerate(norm.get("approval_required") or [])
        if isinstance(a, dict) and isinstance(a.get("timeout"), str) and not _TIMEOUT.fullmatch(a["timeout"].strip())
    ]
    if not bad:
        return []
    return [
        SpecIssue(
            rule_id="AS-107",
            kind="invalid",
            field="approval_required",
            spec_ref=refs.get(("approval_required", bad[0][0]), SourceRef(file=file, line=1)),
            message="approval_required[].timeout must be a number and a unit (s, m, h, d), like '30m' or '4h'"
            + (f" ({len(bad)} entries)" if len(bad) > 1 else "")
            + f"; {SHAPE_HINT}",
            expected="string like 30m | 4h | 2d",
            example=example_yaml("approval_required"),
            items=[f"approval_required[{i}].timeout: {t}" for i, t in bad],
        )
    ]


def _not_needed_issues(spec: Any, refs: dict[PathKey, SourceRef]) -> list[SpecIssue]:
    """agent_id, risk_level, goal and owner can never be not_needed, and at high or critical risk a safety item
    cannot be either: it must be in place (yes) or a gap (no). A spec error, because the answer would silently
    remove something every project, or this risk level, requires."""
    from agenticreq.not_needed import blocked_message

    if spec is None:
        return []
    out = []
    for i, a in enumerate(spec.confirmed):
        why = blocked_message(a.clause, spec.risk_level) if a.answer == "not_needed" else ""
        if why:
            out.append(
                SpecIssue(
                    rule_id="AS-107",
                    kind="invalid",
                    field="confirmed",
                    spec_ref=refs.get(("confirmed", i)),
                    message=why,
                    expected="answer: yes (by, date, expires, evidence) or answer: no (by, date, reason)",
                    example=example_yaml("confirmed"),
                    items=[a.clause],
                )
            )
    return out


RENAMED_ATTESTATIONS = "`attestations:` was renamed to `confirmed:`; rename the key in spec.yml"


def _finish(
    path: Path,
    raw: dict[str, Any],
    refs: dict[PathKey, SourceRef],
    dups: list[tuple[PathKey, SourceRef]],
    findings: list[Finding],
    chain: list[str],
) -> LoadedSpec:
    from agenticreq.loader.issues import fmt_path

    from agenticreq.not_needed import extract

    issues: list[SpecIssue] = [_finding_issue(f) for f in findings]  # extends problems (AS-115)
    raw, not_needed, blocked = extract(raw, refs, str(path))  # `field: not_needed` answers the field
    issues += blocked
    for dpath, dref in dups:
        top = str(dpath[0]) if dpath else ""
        issues.append(
            SpecIssue(
                rule_id="AS-105" if top == "limits" else "AS-107",
                kind="invalid",
                field=top,
                spec_ref=dref,
                message=f"duplicate key {fmt_path(dpath)!r}: YAML keeps only the last value, so the first is lost",
                expected="each key once",
                example=example_yaml(top) if top else "",
                items=[fmt_path(dpath)],
            )
        )
    todos: list[PathKey] = []
    placeholders: list[tuple[PathKey, str]] = []
    stripped = _strip_todos(raw, (), todos, placeholders)
    if placeholders:  # one defect, however many fields: ${OWNER}, ${TECHNICAL_OWNER} ...
        issues.append(placeholder_issue(placeholders, refs.get(placeholders[0][0])))
    data, notes, prose = extract_prose(stripped, frozenset(todos))
    for key, items in prose.items():
        issues.append(prose_issue(key, items, refs.get(items[0][0]) or refs.get((key,))))
    if "attestations" in data:  # the old name of `confirmed:`: stop, never read it silently or ignore it
        data.pop("attestations")
        issues.append(
            SpecIssue(
                rule_id="AS-107",
                kind="invalid",
                field="attestations",
                spec_ref=refs.get(("attestations",)),
                message=RENAMED_ATTESTATIONS,
                expected="confirmed: [{clause, by, date, evidence, expires}]",
                example=example_yaml("confirmed"),
                items=["attestations"],
            )
        )
    pre = precheck(data, lambda k: refs.get((k,)))
    issues += pre
    for i in pre:
        data.pop(i.field, None)  # do not cascade one bad shape into a pile of type errors
    norm = _normalise(data, refs)
    issues += _timeout_issues(norm, refs, str(path))
    errors: list[tuple[dict[str, Any], SourceRef]] = []
    spec = _build_model(norm, refs, str(path), errors, frozenset(todos))
    issues += errors_to_issues(errors)
    issues += _not_needed_issues(spec, refs)

    from agenticreq.models.spec import TIER2_FIELDS, TIER3_FIELDS

    # user overrides of defaulted fields (Tier 2, and the defaulted Tier 3 ones)
    user_tier2 = {k: stripped[k] for k in (*TIER2_FIELDS, *TIER3_FIELDS) if k in stripped and stripped[k] is not None}
    return LoadedSpec(
        path=path,
        raw=raw,
        refs=refs,
        spec=spec,
        todos=todos,
        load_findings=[],
        user_tier2=user_tier2,
        extends_chain=chain,
        issues=issues,
        notes=notes,
        not_needed=not_needed,
    )


def _finding_issue(f: Finding) -> SpecIssue:
    return SpecIssue(
        rule_id=f.rule_id,
        kind="invalid",
        field="extends",
        spec_ref=f.spec_ref,
        message=f.detail,
        expected="a profile name or a path to another spec",
        example=example_yaml("extends"),
        items=[f.evidence],
    )


def _yaml_error(p: Path, e: Exception) -> LoadedSpec:
    mark = getattr(e, "problem_mark", None)
    line = mark.line + 1 if mark is not None else 1
    problem = getattr(e, "problem", None) or str(e).splitlines()[0]
    issue = SpecIssue(
        rule_id="AS-107",
        kind="invalid",
        field="(file)",
        spec_ref=SourceRef(file=str(p), line=line),
        message=f"the file is not valid YAML: {problem} (line {line}). Common causes: a tab for "
        "indentation, an unquoted value containing ': ' or starting with '{' / '[' / '*'.",
        expected="YAML mapping",
        example="agent_id: support-agent\nrisk_level: high",
        items=[str(e).replace("\n", " ")[:200]],
    )
    return LoadedSpec(path=p, raw={}, refs={}, spec=None, issues=[issue])


def load_spec(path: str | Path) -> LoadedSpec:
    p = Path(path)
    findings: list[Finding] = []
    chain: list[str] = []
    try:
        raw, refs, dups = _load_raw_with_extends(p, None, [path_key(p.resolve())], findings, chain)
    except (yaml.YAMLError, ValueError, UnicodeDecodeError) as e:
        return _yaml_error(p, e)
    return _finish(p, raw, refs, dups, findings, chain)


def load_spec_text(text: str, path: str = "<memory>.yml") -> LoadedSpec:
    p = Path(path)
    findings: list[Finding] = []
    chain: list[str] = []
    try:
        raw, refs, dups = _load_raw_with_extends(p, text, [path], findings, chain)
    except (yaml.YAMLError, ValueError) as e:
        return _yaml_error(p, e)
    return _finish(p, raw, refs, dups, findings, chain)
