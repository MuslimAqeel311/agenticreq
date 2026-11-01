"""`agenticreq validate --fix`: rename unknown fields that have one clear match, and rewrite the unambiguous
prose-to-structure cases, leaving the rest as TODO.

Renames: an unknown key is renamed only when the suggestion engine (loader/issues.py) finds exactly one field it
is closest to, in the same mapping (``prohibted`` -> ``prohibited``, ``bounds.maxSteps`` -> ``bounds.max_steps``).
Two equally close fields, no close field, or a field that belongs in another mapping are left for the person.
Only the key on its own line changes; the value, comments and every other line stay as they are.

Prose: only shape is used, never meaning: a sentence is moved, verbatim, under ``note:`` and the structured part
becomes ``TODO`` for the user to write. Nothing is inferred from the words. Each rewrite replaces exactly
the lines of that one top-level field; every other line, comment and line ending is kept as it was.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agenticreq.loader.issues import NAME_LISTS, PROSE_FIELDS, is_list_field, is_name
from agenticreq.shapes import example, example_yaml


@dataclass
class Rewrite:
    field: str
    why: str
    start: int  # first line (0-based)
    end: int  # one past the last line
    text: str  # replacement lines, without the trailing newline


def _one_line(key: str) -> str:
    """The example as one flow-style line, for a `# shape:` hint next to a TODO."""
    return json.dumps(example(key), ensure_ascii=False).replace('"', "")


def _q(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)  # a JSON string is a valid, safely quoted YAML scalar


def _note_lines(note: str | list[str]) -> list[str]:
    if isinstance(note, list):
        return ["  note:"] + [f"    - {_q(n)}" for n in note]
    return [f"  note: {_q(note)}"]


def _todo_block(key: str, note: str | list[str], kept: list[Any] | None = None) -> str:
    lines = [f"{key}:", *_note_lines(note)]
    if is_list_field(key):
        if kept:
            lines.append("  rules:")
            lines += [f"    - {json.dumps(k, ensure_ascii=False)}" for k in kept]
            lines.append(f"    # add the rest in this shape: {_one_line(key)[1:-1]}")
        else:
            lines.append(f"  rules: TODO   # shape: {_one_line(key)}")
    else:
        ex = example(key)
        for k in ex if isinstance(ex, dict) else []:
            lines.append(f"  {k}: TODO   # e.g. {json.dumps(ex[k], ensure_ascii=False)}")
    return "\n".join(lines)


def _plan_field(key: str, value: Any) -> tuple[str, str] | None:
    """(why, replacement) for one top-level field, or None when nothing is unambiguous to rewrite."""
    if isinstance(value, str) and value.strip().upper().startswith("TODO"):
        return None
    if key == "goal" and isinstance(value, str):
        return (
            "goal is a sentence: it becomes goal.statement; done_criteria are yours to write",
            f"goal:\n  statement: {_q(value)}\n  done_criteria: TODO   # checkable statements, e.g. "
            f"{json.dumps(example('goal')['done_criteria'], ensure_ascii=False)}",
        )
    if key == "out_of_scope" and isinstance(value, str):
        return ("out_of_scope is one sentence: it becomes a one-item list", f"out_of_scope:\n  - {_q(value)}")
    if key == "escalation" and isinstance(value, dict) and {"metric", "op", "threshold", "to"} <= set(value):
        return (
            "escalation is one rule: it becomes a one-item list",
            "escalation:\n  - " + json.dumps(value, ensure_ascii=False),
        )
    if key not in PROSE_FIELDS or isinstance(value, dict) and "note" in value:
        return None  # already in note form: nothing left that is unambiguous
    if isinstance(value, str):
        return (f"{key} is prose: kept verbatim as a note; the structure is TODO", _todo_block(key, value))
    if (
        key == "role_boundary"
        and isinstance(value, list)
        and value
        and all(isinstance(v, str) and is_name(v) for v in value)
    ):
        return (
            "role_boundary is a list of tool names: it becomes role_boundary.allowed_tools",
            "role_boundary:\n  allowed_tools: " + json.dumps(value, ensure_ascii=False),
        )
    if key == "limits" and isinstance(value, dict) and any(not isinstance(v, dict) for v in value.values()):
        pairs = [f"{k}: {v}" for k, v in value.items()]
        return (
            "limits are `name: number` pairs, which do not say which tool argument they bound or whether "
            "the number is a max or a min: kept verbatim as a note; the limits are TODO",
            _todo_block(key, pairs),
        )
    if isinstance(value, list) and is_list_field(key):
        prose = [
            v
            for v in value
            if isinstance(v, str)
            and not (key in NAME_LISTS and is_name(v))
            and not v.strip().upper().startswith("TODO")
        ]
        if not prose:
            return None
        kept = [v for v in value if v not in prose]
        field_name = NAME_LISTS.get(key)
        kept_struct = [{field_name: v} if isinstance(v, str) and field_name else v for v in kept]
        return (
            f"{len(prose)} {key} entr{'y is' if len(prose) == 1 else 'ies are'} prose: kept verbatim as a note; "
            "those rules are TODO",
            _todo_block(key, prose, kept_struct),
        )
    return None


def plan(text: str) -> list[Rewrite]:
    node = yaml.compose(text, Loader=yaml.SafeLoader)
    if not isinstance(node, yaml.MappingNode):
        return []
    loader = yaml.SafeLoader("")
    lines = text.splitlines()
    out = []
    for knode, vnode in node.value:
        key = str(loader.construct_object(knode, deep=True))
        value = loader.construct_object(vnode, deep=True)
        got = _plan_field(key, value)
        if got is None:
            continue
        end = vnode.end_mark.line + (1 if vnode.end_mark.column > 0 else 0)
        while end > knode.start_mark.line + 1 and not lines[end - 1].strip():
            end -= 1  # a block value's end mark runs on over blank lines: keep them
        out.append(
            Rewrite(
                field=key, why=got[0], start=knode.start_mark.line, end=max(end, knode.start_mark.line + 1), text=got[1]
            )
        )
    return out


def apply_rewrites(text: str, rewrites: list[Rewrite]) -> str:
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    for r in sorted(rewrites, key=lambda r: r.start, reverse=True):
        lines[r.start : r.end] = r.text.split("\n")
    return newline.join(lines) + (newline if text.endswith(("\n", "\r")) else "")


@dataclass
class Rename:
    old: str  # full dotted name as written (bounds.maxSteps)
    new: str  # the field it is renamed to (bounds.max_steps)
    line: int  # 1-based


@dataclass
class Unrenamed:
    name: str  # full dotted name as written
    near: list[str]  # the candidates (several, or one in another mapping); [] = none
    line: int | None
    taken: bool = False  # the one clear match is already in that mapping: renaming would duplicate it


def _walk_unknown(raw: dict[str, Any], refs: dict[Any, Any]) -> list[tuple[tuple[Any, ...], str]]:
    """(loc, parent path) of every key spec.yml holds that no field of its mapping is called."""
    from agenticreq.loader.issues import field_paths

    paths = field_paths()
    out: list[tuple[tuple[Any, ...], str]] = []

    def walk(node: Any, loc: tuple[Any, ...], parent: str) -> None:
        if isinstance(node, dict):
            if parent not in paths:
                return  # a free-form mapping (data_rules, _derived): any key is allowed
            for k, v in node.items():
                if str(k) not in paths[parent]:
                    out.append(((*loc, k), parent))
                    continue
                child = f"{parent}.{k}" if parent else str(k)
                walk(v, (*loc, k), child if child in paths else child + "[]" if isinstance(v, list) else child)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, (*loc, i), parent if parent.endswith("[]") else parent + "[]")

    walk(raw, (), "")
    return out


def plan_renames(text: str) -> tuple[list[Rename], list[Unrenamed]]:
    """Which unknown keys `--fix` renames (exactly one close field, in the same mapping) and which it leaves."""
    from agenticreq.loader import load_spec_text
    from agenticreq.loader.issues import fmt_path, suggest

    loaded = load_spec_text(text)
    renames: list[Rename] = []
    left: list[Unrenamed] = []
    for loc, parent in _walk_unknown(loaded.raw, loaded.refs):
        name = fmt_path(loc)
        ref = loaded.refs.get(loc)
        line = ref.line if ref is not None else None
        near = suggest(str(loc[-1]), parent)
        same_mapping = len(near) == 1 and (near[0].rsplit(".", 1)[0] if "." in near[0] else "") == parent
        mapping: Any = loaded.raw
        for part in loc[:-1]:
            mapping = mapping[part] if isinstance(mapping, (dict, list)) else None
        taken = same_mapping and isinstance(mapping, dict) and near[0].rsplit(".", 1)[-1] in mapping
        if same_mapping and not taken and line is not None:
            renames.append(Rename(old=name, new=near[0], line=line))
        else:
            left.append(Unrenamed(name=name, near=near, line=line, taken=bool(taken)))
    return renames, left


def apply_renames(text: str, renames: list[Rename]) -> str:
    lines = text.splitlines(keepends=True)
    for r in renames:
        old_key, new_key = r.old.rsplit(".", 1)[-1].split("[")[0], r.new.rsplit(".", 1)[-1]
        i = r.line - 1
        if 0 <= i < len(lines):
            lines[i] = re.sub(rf"(^|[\s{{,\-])({re.escape(old_key)})(\s*:)", rf"\g<1>{new_key}\g<3>", lines[i], count=1)
    return "".join(lines)


@dataclass
class FixResult:
    path: Path
    before: str
    after: str
    rewrites: list[Rewrite]
    renames: list[Rename] = field(default_factory=list)
    unrenamed: list[Unrenamed] = field(default_factory=list)

    def diff(self) -> str:
        return "".join(
            difflib.unified_diff(
                self.before.splitlines(keepends=True),
                self.after.splitlines(keepends=True),
                fromfile=str(self.path),
                tofile=f"{self.path} (fixed)",
            )
        )


def fix_spec(path: Path) -> FixResult:
    raw = path.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    renames, unrenamed = plan_renames(text)
    renamed = apply_renames(text, renames) if renames else text
    rewrites = plan(renamed)
    after = apply_rewrites(renamed, rewrites) if rewrites else renamed
    if bom:
        after = "﻿" + after
        text = "﻿" + text
    return FixResult(path=path, before=text, after=after, rewrites=rewrites, renames=renames, unrenamed=unrenamed)


def write(result: FixResult) -> None:
    # bytes, so the platform does not translate the line endings the file already has
    result.path.write_bytes(result.after.encode("utf-8"))


__all__ = [
    "FixResult",
    "Rename",
    "Rewrite",
    "Unrenamed",
    "apply_renames",
    "apply_rewrites",
    "example_yaml",
    "fix_spec",
    "plan",
    "plan_renames",
    "write",
]
