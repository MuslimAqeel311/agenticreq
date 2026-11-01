"""Spec-file problems in words a person can act on: what is wrong, the exact shape, and what to write.

Shape detection only. Nothing here interprets what a sentence means (no LLM, no keyword guessing): a
string where a mapping or list is required is *prose*; a multi-word string where a tool or source name is
required is *prose*; a ``${VAR}`` is a *placeholder*. Prose is kept as a note and the field stays
undeclared, so it is reported as UNKNOWN and never treated as satisfied.
"""

from __future__ import annotations

import difflib
import functools
import re
from pathlib import Path
from typing import Any

from agenticreq.loader.routes import describe
from agenticreq.models.finding import SourceRef, SpecIssue
from agenticreq.not_needed import SHAPE_HINT
from agenticreq.shapes import example_yaml, shape, shape_of, spec_fields

PathKey = tuple[str | int, ...]

#: fields people write as sentences; prose there becomes a note and the field stays TODO
PROSE_FIELDS = (
    "role_boundary",
    "prohibited",
    "approval_required",
    "limits",
    "escalation",
    "untrusted_inputs",
    "data_rules",
)
#: list fields whose string items are names (tools / sources): a multi-word string is prose, not a name
NAME_LISTS = {"prohibited": "tool", "approval_required": "tool", "untrusted_inputs": "source"}
_NAME = re.compile(r"^[A-Za-z_][\w.:/@-]*$")
_PLACEHOLDER = re.compile(r"\$\{[^}]*\}|\$[A-Z_][A-Z0-9_]*\b")


def is_name(s: str) -> bool:
    return bool(_NAME.match(s.strip()))


def is_placeholder(v: Any) -> bool:
    return isinstance(v, str) and bool(_PLACEHOLDER.search(v))


def _short(v: Any, n: int = 70) -> str:
    s = str(v).replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def fmt_path(loc: PathKey) -> str:
    out = ""
    for p in loc:
        out += f"[{p}]" if isinstance(p, int) else (f".{p}" if out else str(p))
    return out


def _pattern(loc: PathKey) -> str:
    """limits.0.limit, limits.3.limit -> limits[].limit: one pattern for every index."""
    out = ""
    for p in loc:
        out += "[]" if isinstance(p, int) else (f".{p}" if out else str(p))
    return out


def is_list_field(key: str) -> bool:
    return shape(key).startswith("[")


def note_form(key: str) -> str:
    """The same field with the sentence kept: the structure stays TODO until it is written as above."""
    from agenticreq.shapes import example

    if is_list_field(key):
        rest = "  rules: TODO          # the list above"
    else:
        ex = example(key)
        rest = "\n".join(f"  {k}: TODO" for k in (ex if isinstance(ex, dict) else {"value": None}))
    return f'{key}:\n  note: "<your sentence>"   # kept as a comment, never checked\n{rest}'


# ---------------------------------------------------------------- prose / placeholders (before the model)
def extract_prose(
    d: dict[str, Any], todos: frozenset[PathKey] = frozenset()
) -> tuple[dict[str, Any], dict[str, str], dict[str, list[tuple[PathKey, str]]]]:
    """Move prose out of structured fields: returns (data with prose removed, notes, prose per field).

    * ``field: "a sentence"`` -> note; the field becomes undeclared
    * ``field: {note: "...", rules: [...]}`` / ``{note: "...", <keys>}`` -> note; the rest is the value
    * ``prohibited: ["giving investment advice"]`` -> that item is prose (a tool / source name has no spaces)
    """
    notes: dict[str, str] = {}
    prose: dict[str, list[tuple[PathKey, str]]] = {}
    out = dict(d)
    for key in PROSE_FIELDS:
        v = out.get(key)
        if isinstance(v, str):
            notes[key] = v
            prose.setdefault(key, []).append(((key,), v))
            out[key] = None
            continue
        if isinstance(v, dict) and "note" in v:
            note = v.get("note")
            if isinstance(note, list) and all(isinstance(n, str) for n in note):
                note = "; ".join(note)  # itemised prose (what `validate --fix` writes for prose list items)
            if isinstance(note, str):
                notes[key] = note
            rest = {k: x for k, x in v.items() if k != "note" and x is not None}  # TODO sub-keys are None
            if is_list_field(key):
                rest_v: Any = rest.get("rules")
            else:
                rest_v = rest or None
            pending = any(t[:1] == (key,) and len(t) > 1 for t in todos)  # `rules: TODO`: already a TODO
            if rest_v is None and isinstance(note, str) and not pending:
                prose.setdefault(key, []).append(((key,), note))
            out[key] = rest_v
            v = rest_v
        if isinstance(v, list) and is_list_field(key):
            kept = []
            for i, item in enumerate(v):
                # a name list takes `tool_name` shorthand; anywhere else a string item is a sentence
                names_ok = key in NAME_LISTS and isinstance(item, str) and is_name(item)
                if isinstance(item, str) and not names_ok and not item.strip().upper().startswith("TODO"):
                    prose.setdefault(key, []).append(((key, i), item))
                else:
                    kept.append(item)
            if len(kept) != len(v):
                notes[key] = "; ".join(t for _, t in prose[key])
                out[key] = kept or None
    return out, notes, prose


def prose_issue(key: str, items: list[tuple[PathKey, str]], ref: SourceRef | None) -> SpecIssue:
    n = len(items)
    what = (
        "is written as prose"
        if items[0][0] == (key,)
        else f"has {n} entr{'y' if n == 1 else 'ies'} written as prose instead of "
        + (f"{NAME_LISTS[key]} names" if key in NAME_LISTS else "structured entries")
    )
    return SpecIssue(
        rule_id="AS-108" if key == "escalation" else "AS-118",
        kind="prose",
        field=key,
        spec_ref=ref,
        message=f"{key} {what}, so it cannot be verified. The text is kept as a note and {key} stays "
        "undeclared (a gap, not satisfied) until it is written as structure.",
        expected=shape(key),
        example=example_yaml(key) + "\n# or keep the sentence next to it:\n" + note_form(key),
        items=[f"{fmt_path(p)}: {_short(t)}" for p, t in items],
    )


def placeholder_issue(paths: list[tuple[PathKey, str]], ref: SourceRef | None) -> SpecIssue:
    tops = list(dict.fromkeys(str(p[0]) for p, _ in paths))
    known = [t for t in tops if t in spec_fields()]
    return SpecIssue(
        rule_id="AS-101",
        kind="placeholder",
        field=", ".join(tops),
        spec_ref=ref,
        message="agenticreq does not expand environment variables in a spec: "
        + ("this value is a placeholder" if len(paths) == 1 else f"these {len(paths)} values are placeholders")
        + ", so the field stays undeclared (a gap). Write the real value in the spec.",
        expected="string",
        example="\n".join(example_yaml(t) for t in known),
        items=[f"{fmt_path(p)}: {v}" for p, v in paths],
    )


# ---------------------------------------------------------------- structural pre-checks
def precheck(d: dict[str, Any], ref_of: Any) -> list[SpecIssue]:
    """Shapes the normaliser would otherwise mangle into confusing type errors."""
    out: list[SpecIssue] = []
    lim = d.get("limits")
    if isinstance(lim, dict):
        flat = [(k, v) for k, v in lim.items() if not isinstance(v, dict)]
        if flat:
            out.append(
                SpecIssue(
                    rule_id="AS-107",
                    kind="invalid",
                    field="limits",
                    spec_ref=ref_of("limits"),
                    message="limits must be a list of {tool, arg, limit}; these entries are `name: number` pairs, which "
                    "say neither which tool argument they bound nor whether the number is a max or a min. "
                    "Each entry below names where it belongs, from its key name only; " + SHAPE_HINT,
                    expected=shape("limits"),
                    example=example_yaml("limits"),
                    items=[f"limits.{describe(k, v)}" for k, v in flat],
                )
            )
    return out


# ---------------------------------------------------------------- pydantic errors -> issues
def _shape_at(loc: PathKey) -> str:
    """The expected shape at a nested location, walked through the model's annotations."""
    from typing import get_args, get_origin

    from pydantic import BaseModel

    from agenticreq.models.spec import AgentSpec
    from agenticreq.shapes import _non_none

    ann: Any = AgentSpec
    for part in loc:
        alts = _non_none(ann)
        nxt = None
        for a in alts:
            if isinstance(part, int) and get_origin(a) is list:
                nxt = get_args(a)[0]
            elif isinstance(part, str) and isinstance(a, type) and issubclass(a, BaseModel):
                f = next((f for n, f in a.model_fields.items() if part in (n, f.alias)), None)
                nxt = f.annotation if f else None
            elif isinstance(part, str) and get_origin(a) is dict:
                nxt = get_args(a)[1]
            if nxt is not None:
                break
        if nxt is None:
            return ""
        ann = nxt
    return shape_of(ann)


def _got(v: Any) -> str:
    if isinstance(v, str):
        return f"the text {_short(v, 50)!r}"
    if isinstance(v, bool):
        return f"{v!r}"
    if isinstance(v, (int, float)):
        return f"the number {v}"
    if isinstance(v, list):
        return f"a list of {len(v)}"
    if isinstance(v, dict):
        return "a mapping {" + ", ".join(list(map(str, v))[:4]) + "}"
    return type(v).__name__


# ------------------------------------------------------------------ an unknown field name: the closest real one
#: words that say nothing about which field is meant
_FILLER = {"the", "a", "an", "agent", "allow"}
#: how close a normalised name must be to count (difflib ratio)
CUTOFF = 0.75


def normalise(name: str) -> str:
    """maxSteps / max-steps / Max_Steps -> max_step: lowercase snake_case, filler words dropped, a trailing plural
    s dropped. Both the unknown name and every real one go through this before they are compared."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name.strip())
    words = [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w and w not in _FILLER]
    if words and len(words[-1]) > 3 and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
    return "_".join(words)


@functools.lru_cache(maxsize=1)
def field_paths() -> dict[str, tuple[str, ...]]:
    """Every field spec.yml can hold, by the path of the mapping it sits in: "" -> top-level names,
    "bounds" -> max_steps, ..., "approval_required[]" -> the fields of one list entry."""
    import types
    import typing

    from pydantic import BaseModel

    from agenticreq.models.spec import AgentSpec

    out: dict[str, tuple[str, ...]] = {}

    def models_in(ann: Any) -> list[tuple[type[BaseModel], bool]]:
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            return [(ann, False)]
        origin = typing.get_origin(ann)
        args = typing.get_args(ann)
        if origin in (list, tuple, set):
            return [(m, True) for a in args for m, _ in models_in(a)]
        if origin in (typing.Union, types.UnionType):
            return [x for a in args for x in models_in(a)]
        return []

    def walk(model: type[BaseModel], path: str, depth: int) -> None:
        names = tuple(f.alias or n for n, f in model.model_fields.items())
        out[path] = tuple(dict.fromkeys(out.get(path, ()) + names))
        if depth > 3:
            return
        for n, f in model.model_fields.items():
            for sub, is_list in models_in(f.annotation):
                walk(sub, f"{path + '.' if path else ''}{f.alias or n}{'[]' if is_list else ''}", depth + 1)

    walk(AgentSpec, "", 0)
    return out


def _aka() -> dict[str, tuple[str, ...]]:
    """Full field name -> the other names in its params catalog entry."""
    from agenticreq.params import PARAMS

    return {full: words for p in PARAMS for full, words in p.also_called()}


def _score(unknown: str, candidate: str, alias: bool = False) -> float:
    a, b = normalise(unknown), normalise(candidate)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta, tb = a.split("_"), b.split("_")
    if alias and set(tb) <= set(ta) and len(ta) - len(tb) <= 1:  # forbidden_tools -> the alias forbidden
        return 0.95
    common = set(ta) & set(tb)
    if common and set(ta) - common and set(tb) - common:
        # shared words say nothing about which field is meant: compare the words that differ
        # (prohibted_tools vs permitted_tools is prohibted vs permitted, not a near-match)
        a = "_".join(t for t in ta if t not in common)
        b = "_".join(t for t in tb if t not in common)
    return difflib.SequenceMatcher(None, a, b).ratio()


def suggest(name: str, parent: str = "") -> list[str]:
    """The closest real field(s) for an unknown ``name`` found inside ``parent`` ("" = top level), as full dotted
    names. Matches the name itself (typos, maxSteps, a plural), then the "also called" words of the params catalog.
    Several when they are equally close; [] when nothing is close enough. Never more than one guess is chosen."""
    paths = field_paths()
    aka = _aka()
    if parent:
        pool = [f"{parent}.{n}" for n in paths.get(parent, ())]
    else:  # a top-level name may belong inside a mapping (max_steps -> bounds.max_steps)
        pool = [f"{p}.{n}" if p else n for p, names in paths.items() for n in names]
    scores: dict[str, float] = {}
    for full in pool:
        last = full.rsplit(".", 1)[-1]
        best = max([_score(name, last), *(_score(name, w, alias=True) for w in aka.get(full.replace("[]", ""), ()))])
        scores[full] = best
    top = max(scores.values(), default=0.0)
    if top < CUTOFF:
        return []
    near = [f for f, sc in scores.items() if sc == top]
    shallow = min(f.count(".") for f in near)  # a top-level name means a top-level field before a nested one
    return sorted(f for f in near if f.count(".") == shallow)


def _quote(names: list[str]) -> str:
    return " or ".join(f"`{n}`" for n in names)


def unknown_field_message(loc: PathKey, ref: SourceRef | None, near: list[str]) -> str:
    """Unknown field `max_iterations` in spec.yml (line 12). Did you mean `bounds.max_steps`? Fix it, or run: ..."""
    where = f"{Path(ref.file).name} (line {ref.line})" if ref is not None and ref.file else "spec.yml"
    head = f"Unknown field `{fmt_path(loc)}` in {where}."
    if near:
        return f"{head} Did you mean {_quote(near)}? Fix it, or run: agenticreq validate --fix"
    return (
        f"{head} It is not a field agenticreq knows, so it would not be checked: remove it, or see every field "
        "with: agenticreq params"
    )


def parent_path(loc: PathKey) -> str:
    """("approval_required", 0, "timout") -> "approval_required[]"; ("bounds", "maxSteps") -> "bounds"."""
    out = ""
    for part in loc[:-1]:
        out = out + "[]" if isinstance(part, int) else (f"{out}.{part}" if out else str(part))
    return out


def _human(err: dict[str, Any], loc: PathKey) -> str:
    where = fmt_path(loc) or "the spec"
    t = err["type"]
    inp = err.get("input")
    expected = _shape_at(loc)
    if t == "extra_forbidden":
        parent = loc[:-1]
        name = str(loc[-1])
        options = spec_fields() if not parent else []
        near = difflib.get_close_matches(name, options, n=1, cutoff=0.75)
        return f"unrecognised key {where!r}: it would not be checked" + (
            f" (did you mean {near[0]!r}?)" if near else ""
        )
    if t == "missing":
        return f"{where} is required here"
    if t in ("literal_error", "enum"):
        return f"{where} must be one of {expected}; got {_got(inp)}"
    if t in ("dict_type", "model_type", "model_attributes_type"):
        return f"{where} must be a mapping {expected}; got {_got(inp)}"
    if t == "list_type":
        return f"{where} must be a list {expected}; got {_got(inp)}"
    if t in (
        "string_type",
        "int_type",
        "float_type",
        "bool_type",
        "int_parsing",
        "float_parsing",
        "bool_parsing",
        "date_type",
        "date_parsing",
        "date_from_datetime_parsing",
    ):
        return f"{where} must be {expected or t.split('_')[0]}; got {_got(inp)}"
    if t in ("greater_than", "greater_than_equal", "less_than_equal", "less_than"):
        # a number that cannot be a limit (0 or below, a confidence outside 0..1) is a shape error; any positive
        # number is the user's choice at every risk level
        return f"{where} must be {err.get('msg', '').removeprefix('Input should be ')}; got {_got(inp)}"
    return f"{where}: expected {expected or 'a different value'}; got {_got(inp)}"


def errors_to_issues(errors: list[tuple[dict[str, Any], SourceRef]]) -> list[SpecIssue]:
    """One issue per (field, location pattern, error type): limits[0..4].limit is ONE issue with 5 items."""
    groups: dict[tuple[str, str, str], list[tuple[dict[str, Any], SourceRef]]] = {}
    for err, ref in errors:
        loc = tuple(err["loc"])
        key = (str(loc[0]) if loc else "", _pattern(loc), err["type"])
        groups.setdefault(key, []).append((err, ref))
    out = []
    for (top, pattern, etype), errs in groups.items():
        first_err, first_ref = errs[0]
        loc0 = tuple(first_err["loc"])
        near: list[str] = []
        if etype == "extra_forbidden":
            near = suggest(str(loc0[-1]), parent_path(loc0))
            msg = unknown_field_message(loc0, first_ref, near)
        else:
            msg = _human(first_err, loc0)
        if len(errs) > 1:
            msg = msg.replace(fmt_path(loc0), pattern, 1) + f" ({len(errs)} entries)"
        if etype != "extra_forbidden":
            msg += f"; {SHAPE_HINT}"
        rule = "AS-117" if etype == "extra_forbidden" else "AS-107"
        known = top in spec_fields()
        out.append(
            SpecIssue(
                rule_id=rule,
                kind="invalid",
                field=top,
                spec_ref=first_ref,
                message=msg,
                expected=_shape_at(loc0) if etype != "extra_forbidden" else "",
                example=example_yaml(top) if known else "",
                items=[f"{fmt_path(tuple(e['loc']))}: {_short(e.get('input'))}" for e, _ in errs],
                did_you_mean=near,
            )
        )
    return out
