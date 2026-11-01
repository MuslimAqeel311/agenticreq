# How tools are read, and what a mock run reports

## The guiding rule

When agenticreq is not sure, the result goes to **Check with a mock run**, with the reason. It is never a
**Spec vs code gap** (that would be a false alarm) and never **Passed** (that would be a false pass). A gap needs
clear evidence in the code.

## Limits: how a cap on a tool is found

A tool is capped when its body, or a project helper it calls, limits what it returns or takes. These forms are read
(`src/agenticreq/ast/limits.py`):

| form | example |
|---|---|
| slicing | `rows[:200]`, `text[:5000]`, `items[-50:]` |
| `itertools.islice` | `islice(rows, 200)` |
| a counter that stops the loop | `if len(rows) >= 200: break` (or `return`), `for i, r in enumerate(...): if i >= 40: break` |
| `min` / `max` | `n = max(1, min(int(max_results), 10))` |
| a size check that refuses | `if len(content) > MAX_FILE_BYTES: raise ...` (or returns an error) |
| SQL | `LIMIT 25` in a string, an f-string or as a query parameter (`LIMIT ?`) |
| ORM / pandas | `.limit(20)`, `.head(12)`, `.nlargest(7, "x")`, `.fetchmany(50)`, `resp.read(MAX_BYTES)` |
| a capping keyword | `limit=`, `max_results=`, `top_k=`, `page_size=`, `per_page=`, `n=`, `k=` (also in a `params={...}` dict) |
| parameter constraints | `Field(le=50)`, `Field(max_length=200)`, JSON Schema `maximum`, `maxItems`, `maxLength` |
| a project helper | `truncate(text, 6000)` where `truncate` slices its own parameter |

Numbers are read through module constants (`MAX_ROWS = 200`), class attributes (`self.MAX`), constants imported
from another project module, and local variables. A slice used only inside a test (`"<html" in body[:500]`) caps
nothing and is ignored. A cap set by an argument is reported with its default: "capped by the limit argument
(default 5)".

The report gives the cap and where it is: `list_files capped at 200 (counter, multi_agent/tools.py:193)`. When the
value is read from the environment or a settings object (`int(os.getenv("MAX_ROWS"))`, `settings.max_rows`), the
cap exists but its value is not in the code: the line is **Check with a mock run: limit set from env MAX_ROWS**. A
tool with no cap at all stays a gap: `search_everything can return unbounded output`.

## Tool effects: READ, WRITE or UNCLEAR

Each tool is classified by what its body does, following calls into project functions (`src/agenticreq/ast/effects.py`).
The tool's name (`save_`, `delete_`, `get_`) is only a weak hint, used when the body cannot be read; it never
overrides what the body shows.

- **WRITE**: `open(..., "w" / "a" / "x" / "+")`, `Path.write_text / write_bytes / touch / unlink`, `os.remove /
  rename / makedirs`, `shutil` copy / move / rmtree, a program started (`subprocess`, `os.system`), HTTP POST / PUT /
  PATCH / DELETE (requests, httpx, aiohttp, urllib with data), SQL INSERT / UPDATE / DELETE / DROP, ORM `add / commit /
  save / delete` (and their async forms), sending mail or messages, cloud writes (`put_object`, `upload_file`), an
  attribute set on an injected state object (`state.seat = seat`), MCP `readOnlyHint: false` / `destructiveHint: true`.
- **READ**: `open(...)` with no mode or `"r"`, `Path.read_text`, `os.walk / listdir / stat`, HTTP GET, SQL SELECT, MCP
  `readOnlyHint: true`. A body with no I/O at all (math, string processing, parsing, a value it returns) is READ with
  the reason "no side effects found". `mkdir(..., exist_ok=True)` only makes sure a folder exists and is not a write.
  Memory inside the process (a cache dict, a "loaded" flag) is not a write either.
- **UNCLEAR**: the body calls something agenticreq cannot follow (a function of a library it does not know, such as
  `yf.Ticker(...)` or `stripe.Charge.create(...)`), or its body is only a placeholder (`raise`, `pass`). The reason
  names that call and its line, and any result that depends on it goes to **Check with a mock run**.

This classification is what every rule that needs "a write tool" uses: approval (AS-212), write safety and
idempotency (AS-211), dry run (AS-227) and the trust boundary (AS-401).

### Declaring what a tool does: `tool_effects:`

```yaml
tool_effects: {read_file: read, save_report: write}
```

A declaration decides a tool whose body cannot be followed. When the body contradicts it (declared `read`, but the
body writes), the report shows a gap with the line that writes (AS-232):
`spec.yml declares read_file read, but it writes: .write_text() at tools.py:220.`

## `allow_write:` in mocks.yml

```yaml
allow_write: [workspace/]   # paths relative to the project folder
```

- An allowed folder includes everything under it; a single file can be listed too.
- Paths are compared after turning `\` and `/` alike, resolving `..` and symlinks, and ignoring case on Windows. A
  file outside an allowed path is blocked even when it is reached through `../` or a symlink.
- A write inside an allowed path is allowed and listed in the summary (`wrote workspace/final_answer.md, 2 KB`); it
  does not count as a change of the project's files.
- A write outside is blocked. The run goes on if the agent handles the error, and the summary says
  `blocked write to report.md; to allow it, add it under allow_write: in mocks.yml`.
- Without `allow_write:`, only `.agenticreq/` and the system temp folder may be written (inside the project folder,
  even when the project itself is under the temp folder, only `allow_write:` paths and `.agenticreq/` count).
- `agenticreq mocks` writes `allow_write: []` with a comment, and pre-fills the folders the code spells out in its
  write calls (`open("logs/run.log", "a")`, `Path(OUT) / name`, `os.getenv("DIR", "workspace")`). A folder built
  from a settings object is not followed: add it by hand.

## Mock-run failure messages

Every failed or empty run exits 1 with one sentence: what was captured, where the run stopped (file:line in your
project), why, and the next step. A run that captured some calls and then failed keeps them in
`.agenticreq/captures.json`. A clean run exits 0 with no error text.

| what happened | message |
|---|---|
| the entry function or module does not exist | `mock-run captured 0 model calls. The entry point agent.py:nope was not found (KeyError: 'nope'). Fix entry: in mocks.yml (module:function or file.py:function), then run: agenticreq mock-run` |
| a dependency is missing | `mock-run captured 0 model calls. The run stopped at agent.py:1 with ModuleNotFoundError: No module named 'x': a dependency is missing in <python>. Install x there (or set python: in mocks.yml to an environment that has the agent's dependencies), then run: agenticreq mock-run` |
| the agent crashed before calling the model | `mock-run captured 0 model calls. The run stopped at agent.py:3 with KeyError: 'model'. Fix the error, then run: agenticreq mock-run` |
| the SDK rejected the request before sending it | `mock-run captured 0 model calls. The run stopped at agents.py:43 with ValueError: temperature is not supported ...: the langchain_anthropic SDK rejected the request before sending it. Fix the value or the model, then run: agenticreq mock-run` |
| the mock model could not answer a request | `... The run stopped at agent.py:9 with BadRequestError: 400, because the mock model could not answer the <provider> request (...). Change that request (or the model it names), then run: agenticreq mock-run` |
| an HTTP client the mock run cannot capture | `mock-run captured 0 of 1 model calls. 1 call(s) went to api.openai.com through an HTTP client the mock run cannot capture (only httpx / httpx2 are captured ...). Use such a client, or replace that call with patch: in mocks.yml, then run: agenticreq mock-run` |
| a timeout | `mock-run captured 1 model call before the 120s limit stopped run-0 (last step: model call 1, tools offered: ...). The agent may wait for input or loop. Raise --timeout, or fix what it waits for, then run: agenticreq mock-run` |
| no model call at all | `mock-run captured 0 model calls. The entry agent.py:main returned without calling a model (a cached answer, or not the function that runs the agent). Check entry: and args: in mocks.yml, then run: agenticreq mock-run` |
| a retry loop hid the first error | `... with RuntimeError: gave up after 3 attempts. The first error was ConnectionError: refused by proxy (a retry raised the one above). Fix the error, then run: agenticreq mock-run` |
| a partial run | `mock-run captured 4 model calls, then stopped. The run stopped at agent.py:7 with RuntimeError: tool failed. Fix the error, then run: agenticreq mock-run` |

## After a mock run

A line stays under **Check with a mock run** only when editing mocks.yml could settle it (a tool to add to `order:`,
`args:` that reach an agent), and it says what to change. Otherwise it moves: a number in spec.yml with no limit in
code -> gap; something agenticreq cannot read -> sign-off; an approval gate inside a mocked tool -> sign-off. Running
mock-run twice with nothing changed gives the same report.

## Names and runtime agents

Names are compared after `names.norm` (lowercase; spaces, hyphens, dots to `_`). A runtime agent whose prompt opening
holds no text written in the project is the framework's own call (`_derived.injected_by_framework`); one with project
text but no name in code is a sign-off (`confirmed:` clause `runtime_agents`).

## Sub-fields only a person can confirm

The code cannot show `role_boundary.capabilities`, `role_boundary.data_access`, `untrusted_inputs[].kind`,
`output_size_limits.max_items`, `write_safety.compensating_action` or `agent_registry[].role`. Each one a spec writes
becomes a plain question under Human sign-off, for example "Does the agent only read the data listed in
role_boundary.data_access? Confirm under confirmed: (role_boundary.data_access)." `agenticreq check --coverage` lists
every parameter.
