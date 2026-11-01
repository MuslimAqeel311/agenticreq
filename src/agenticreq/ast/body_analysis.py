"""Static analysis of a tool function's own body: side effects, egress guards, fail-open handlers, bounds.

Moved out of ``extractor.py`` unchanged; each function takes the extractor as ``ex``.
"""

from __future__ import annotations

import ast
from agenticreq.ast.classify import BOUND_PARAMS
from agenticreq.ast.schema import _name
from agenticreq.ast.schema import literal
from agenticreq.ast.symbols import FuncDef
from typing import Literal
from agenticreq.ast.extract_support import (
    _ALLOWLIST_NAME,
    _ERROR_TARGET,
    _ERROR_WORDS,
    _EXEC_CALLS,
    _EXEC_MODULES,
    _FRESH_ID_CALLS,
    _GUARD_NAME,
    _HOSTISH,
    _HTTP_CALLS,
    _HTTP_MODULES,
    _LOG_CALLS,
    _RECORD_WRITES,
    _RETRY_NAME,
    _SELF_ATTRS,
    _caps_results,
    _host_controlled,
    _kw,
    _str,
    _unbounded_read,
    _url_vars,
)


def _non_idempotent(fn: FuncDef, params: set[str]) -> bool:
    """Evidence in the body that running the tool twice repeats its side effect.

    * a fresh id is minted (``uuid4()``, ``random.choices``) — unless it is only the fallback of
      reused state (``state.case_id or new_id()``: the second run finds the first id);
    * a record is added to something outside the function (``store.aput``, ``self.rows.append``,
      ``db.insert``, ``INSERT INTO``) — receivers that are locals of the tool don't count.
    """
    guarded: set[int] = set()
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.BoolOp)
            and isinstance(n.op, ast.Or)
            and isinstance(n.values[0], (ast.Attribute, ast.Subscript))
        ):
            for v in n.values[1:]:
                guarded |= {id(x) for x in ast.walk(v)}
    for n in ast.walk(fn):
        if not isinstance(n, ast.Call) or id(n) in guarded:
            continue
        fname = _name(n.func)
        if fname in _FRESH_ID_CALLS:
            return True
        if isinstance(n.func, ast.Attribute) and fname in _RECORD_WRITES:
            recv = n.func.value
            if isinstance(recv, ast.Attribute) or (isinstance(recv, ast.Name) and recv.id in params):
                return True
        if fname in ("execute", "executemany") and n.args:
            sql = _str(n.args[0]) or ""
            if sql.lstrip().upper().startswith("INSERT"):
                return True
    return False


def _body_features(fn: FuncDef | None) -> list[str]:
    if fn is None:
        return []
    params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    feats: set[str] = set()
    if _non_idempotent(fn, params):
        feats.add("non_idempotent")
    url_vars = _url_vars(fn, params)
    dict_vars = {
        t.id: x.value
        for x in ast.walk(fn)
        if isinstance(x, ast.Assign) and isinstance(x.value, ast.Dict)
        for t in x.targets
        if isinstance(t, ast.Name)
    }
    for n in ast.walk(fn):
        if isinstance(n, (ast.If, ast.IfExp, ast.While)) and any(
            isinstance(x, ast.Name)
            and x.id in params
            and x.id.lower().replace("_", "")
            in ("dryrun", "preview", "simulate", "validateonly", "planonly", "confirm")
            for x in ast.walk(n.test)
        ):
            feats.add("dry_run_branch")  # the body actually behaves differently in a dry run
        if (
            isinstance(n, ast.Return)
            and n.value is not None
            and any(
                isinstance(x, ast.Subscript) and isinstance(x.slice, ast.Slice) and x.slice.upper is not None
                for x in ast.walk(n.value)
            )
        ):
            feats.add("bounded_output")  # return results[:n], "\n".join(f"- {o}" for o in objects[:20])
        if (
            isinstance(n, ast.Subscript)
            and isinstance(n.slice, ast.Slice)
            and n.slice.upper is not None
            and isinstance(n.value, ast.Name)
            and n.value.id in params
        ):
            feats.add("bounded_output")  # queries = queries[:max_parallel]: the work, and so the output, is capped
        if not isinstance(n, ast.Call):
            continue
        fname = _name(n.func)
        owner = _name(n.func.value) if isinstance(n.func, ast.Attribute) else ""
        if fname in ("exec", "eval") or (fname in _EXEC_CALLS and owner in _EXEC_MODULES):
            feats.add("exec")
        if fname in _HTTP_CALLS and (owner in _HTTP_MODULES or fname == "urlopen"):
            if fname in ("post", "put", "patch", "delete"):
                feats.add("non_idempotent")  # a remote write: retried, it is sent again
            first = n.args[0] if n.args else _kw(n, "url")
            if first is not None and _host_controlled(first, url_vars):
                feats.add("url_from_arg")
            feats.add("http")
        if _unbounded_read(n):
            feats.add("unbounded_read")
        if any(k.arg in BOUND_PARAMS for k in n.keywords if k.arg):
            feats.add("bounded_output")  # TavilySearch(max_results=...), db.query(limit=10)
        if any(
            _caps_results(dict_vars.get(a.id) if isinstance(a, ast.Name) else a)
            for a in [*n.args, *(k.value for k in n.keywords)]
        ):
            feats.add("bounded_output")  # requests.get(url, params={"limit": 5}), get_json(url, {"hitsPerPage": n})
        if fname in ("open", "write_text", "write_bytes", "unlink", "rmtree", "remove"):
            feats.add(
                "fs_write"
                if fname != "open"
                or any(
                    isinstance(a, ast.Constant) and isinstance(a.value, str) and any(m in a.value for m in "wa+")
                    for a in n.args[1:2]
                )
                else "fs_read"
            )
            if any(isinstance(x, ast.Name) and x.id == "__file__" for a in n.args[:1] for x in ast.walk(a)) or (
                isinstance(n.func, ast.Attribute)
                and any(isinstance(x, ast.Name) and x.id == "__file__" for x in ast.walk(n.func.value))
            ):
                if fname != "open" or "fs_write" in feats:
                    feats.add("self_modify")  # open(__file__, "w"), Path(__file__).write_text(...)
        if fname in _LOG_CALLS and owner and ("log" in owner.lower() or owner == "logging"):
            feats.add("logs")
    for n in ast.walk(fn):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Attribute) and t.attr in _SELF_ATTRS for t in targets):
                feats.add("self_modify")  # agent.instructions = ..., self.system_prompt += ...
        elif (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("append", "extend", "insert")
            and isinstance(n.func.value, ast.Attribute)
            and n.func.value.attr == "tools"
        ):
            feats.add("self_modify")  # agent.tools.append(new_tool)
        elif isinstance(n, ast.ExceptHandler) and _fails_open(n):
            feats.add("fail_open")
    guard = _egress_guarded(fn, params) if "url_from_arg" in feats else None
    if guard == "dominating":
        feats.add("egress_guard")
    elif guard == "partial":
        feats.add("egress_guard_partial")  # a check exists, but a fetch can run before / around it
    if "unbounded_read" in feats and any(
        isinstance(x, ast.Call)
        and _name(x.func) in ("fetchone", "fetchmany", "first", "one", "one_or_none", "scalar", "scalar_one", "get")
        for x in ast.walk(fn)
    ):
        feats.discard("unbounded_read")  # a single row / a page is taken from the result
    return sorted(feats)


def _egress_guarded(fn: FuncDef, params: set[str]) -> Literal["dominating", "partial"] | None:
    """The body checks the model-supplied URL before using it and stops when the check fails:

    * ``if urlparse(url).hostname not in ALLOWED: raise`` / ``assert host in ALLOWED``
    * ``if not url.startswith(("https://a/", ...)): return "refused"`` / ``re.fullmatch(P, url)``
    * a validator called on it: ``_check_host(url)``, ``url = ensure_allowed(url)``
    Locals derived from the parameter (``host = urlparse(url).netloc``) count as the parameter.

    ``dominating``: the check is a top-level statement that runs before any statement that fetches, so
    every path to the HTTP call passes it. ``partial``: a check exists but not in that position (inside a
    branch / try / loop, or after a fetch). None: no check.
    """
    tainted = set(params)
    for _ in range(3):  # follow host = urlparse(url).netloc; domain = host.split(".") ...
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and any(
                isinstance(x, ast.Name) and x.id in tainted for x in ast.walk(n.value)
            ):
                tainted |= {t.id for t in n.targets if isinstance(t, ast.Name)}

    def uses(node: ast.AST) -> bool:
        return any(isinstance(x, ast.Name) and x.id in tainted for x in ast.walk(node))

    def allowlist(node: ast.AST) -> bool:
        """ALLOWED_HOSTS, settings.trusted_domains, {"api.x.com", ...}, ("https://api.x.com/",) — not `cache`."""
        if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
            return bool(node.elts) and all(_HOSTISH.search(_str(e) or "") for e in node.elts)
        if isinstance(node, ast.Constant):
            return bool(_HOSTISH.search(_str(node) or ""))
        return bool(_ALLOWLIST_NAME.search(_name(node)))

    def is_check(test: ast.AST) -> bool:
        if not uses(test):
            return False
        for x in ast.walk(test):
            if (
                isinstance(x, ast.Compare)
                and any(isinstance(o, (ast.In, ast.NotIn)) for o in x.ops)
                and uses(x.left)
                and any(allowlist(c) for c in x.comparators)
            ):
                return True
            if isinstance(x, ast.Call):
                name = _name(x.func)
                if name in ("startswith", "endswith") and x.args and allowlist(x.args[0]):
                    return True  # url.startswith(("https://api.x.com/",)), host.endswith(".corp.com")
                if name in ("match", "fullmatch") and uses(x):
                    return True  # re.fullmatch(ALLOWED_URL, url)
                if _GUARD_NAME.search(name) and any(uses(a) for a in x.args):
                    return True  # if not is_allowed(url): ...
        return False

    def stops(body: list[ast.stmt]) -> bool:
        return any(isinstance(x, (ast.Raise, ast.Return)) for b in body for x in ast.walk(b))

    def guard(n: ast.AST) -> bool:
        if isinstance(n, ast.If) and is_check(n.test) and (stops(n.body) or stops(n.orelse)):
            return True
        if isinstance(n, ast.Assert) and is_check(n.test):
            return True
        value = n.value if isinstance(n, (ast.Expr, ast.Assign)) else None
        return (
            isinstance(value, ast.Call)
            and bool(_GUARD_NAME.search(_name(value.func)))
            and any(uses(a) for a in [*value.args, *(k.value for k in value.keywords)])
        )
        # validate_url(url) / url = ensure_allowed(url): a validator that raises

    def fetches(st: ast.stmt) -> bool:
        return any(
            isinstance(c, ast.Call)
            and _name(c.func) in _HTTP_CALLS
            and (
                (_name(c.func.value) if isinstance(c.func, ast.Attribute) else "") in _HTTP_MODULES
                or _name(c.func) == "urlopen"
            )
            and uses(c)
            for c in ast.walk(st)
        )

    for st in fn.body:
        if guard(st):
            return "dominating"
        if fetches(st):
            break  # the URL is fetched before any top-level check
    return "partial" if any(guard(n) for n in ast.walk(fn)) else None


def _fails_open(h: ast.ExceptHandler) -> bool:
    """A broad handler (bare / Exception / BaseException) that neither re-raises nor reports the failure:
    the tool carries on as if it had succeeded."""
    broad = h.type is None or _name(h.type) in ("Exception", "BaseException")
    if not broad:
        return False
    body = ast.Module(body=h.body, type_ignores=[])

    def carries_error(e: ast.AST | None) -> bool:
        """The value tells the caller something failed: the exception, an error message, False."""
        if e is None:
            return False
        for x in ast.walk(e):
            if h.name and isinstance(x, ast.Name) and x.id == h.name:
                return True
            if isinstance(x, ast.Constant) and (
                x.value is False or (isinstance(x.value, str) and _ERROR_WORDS.search(x.value))
            ):
                return True
        return False

    def empty(e: ast.AST | None) -> bool:
        """None / "" / [] / {} / "ok": looks like 'nothing found' or success to the agent."""
        if e is None:
            return True
        if isinstance(e, ast.Constant):
            return e.value in (None, "", 0) or isinstance(e.value, str)  # a fixed non-error string
        if isinstance(e, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            return not (e.elts if not isinstance(e, ast.Dict) else e.keys)
        # LangGraph: return Command(goto=END, ...) / END ends the run as if it had finished
        return any(isinstance(x, ast.Name) and x.id == "END" for x in ast.walk(e))

    for x in ast.walk(body):
        if isinstance(x, ast.Raise):
            return False
        if isinstance(x, ast.Return):
            if carries_error(x.value) or not empty(x.value):
                return False  # reported to the caller, or a real fallback value (variable / call)
        # recorded where the run can see it: state["error"] = str(e), exception = e, errors.append(e)
        if isinstance(x, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and carries_error(x.value):
            return False
        if isinstance(x, (ast.Assign, ast.AnnAssign)) and any(  # error_trace = traceback.format_exc()
            _ERROR_TARGET.search(_name(t)) for t in (x.targets if isinstance(x, ast.Assign) else [x.target])
        ):
            return False
        # a retry loop counts its attempts and tries again (and gives up with an error after N)
        if isinstance(x, ast.AugAssign) and _RETRY_NAME.search(_name(x.target)):
            return False
        if (
            isinstance(x, ast.Call)
            and _name(x.func) not in _LOG_CALLS | {"print", "format"}
            and any(
                h.name and isinstance(a, ast.Name) and a.id == h.name
                for arg in [*x.args, *(k.value for k in x.keywords)]
                for a in ast.walk(arg)
            )
            and isinstance(x.func, ast.Attribute)
            and x.func.attr in ("append", "add", "update", "put", "set", "send", "emit", "record", "report")
        ):
            return False
    # logging / printing tells an operator, not the agent: the run still continues on a failed step
    return True


def _body_bounds(fn: FuncDef | None) -> dict[str, list[float]]:
    """`if amount > 5000: raise` style checks — arg -> numbers it is compared against."""
    if fn is None:
        return {}
    params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    out: dict[str, list[float]] = {}
    for n in ast.walk(fn):
        if isinstance(n, ast.Compare):  # `limit > 10`, and chained `1 <= limit <= 10`
            operands = [n.left, *n.comparators]
            for left, right in zip(operands, operands[1:]):
                for a, b in ((left, right), (right, left)):
                    if isinstance(a, ast.Name) and a.id in params:
                        ok, v = literal(b)
                        if ok and isinstance(v, (int, float)) and not isinstance(v, bool):
                            out.setdefault(a.id, []).append(float(v))
        elif isinstance(n, ast.Call) and _name(n.func) in ("min", "max"):
            names = [a.id for a in n.args if isinstance(a, ast.Name) and a.id in params]
            nums = [literal(a)[1] for a in n.args if literal(a)[0]]
            for nm in names:
                out.setdefault(nm, []).extend(float(x) for x in nums if isinstance(x, (int, float)))
    return out
