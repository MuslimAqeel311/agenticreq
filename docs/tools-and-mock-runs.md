# Tools and mock runs

**Rule of thumb:** when agenticreq is not sure, the result goes to **Check with a mock run** with the reason. It is
never a gap (a false alarm) and never a pass (a false pass). A gap needs clear evidence in the code.

## Caps on a tool

A tool is capped when its body, or a project helper it calls, limits what it returns or takes. Read forms
(`src/agenticreq/ast/limits.py`): slicing (`rows[:200]`), `islice`, a counter that stops a loop, `min`/`max`, a size
check that refuses, SQL `LIMIT`, ORM and pandas limits (`.limit(20)`, `.head(12)`), capping keywords (`limit=`,
`max_results=`, `top_k=`, `page_size=`), parameter constraints (`Field(le=50)`, JSON Schema `maximum`), and project
helpers such as `truncate(text, 6000)`. Numbers are followed through constants, class attributes and imports.

The report gives the cap and its place (`list_files capped at 200 (counter, tools.py:193)`). A value read from the
environment (`int(os.getenv("MAX_ROWS"))`) exists but is not in the code, so the line goes to Check with a mock run.
A tool with no cap stays a gap.

## Tool effects: READ, WRITE or UNCLEAR

Each tool is classified by what its body does, following calls into project functions (`ast/effects.py`). The tool's
name is only a weak hint when the body cannot be read.

- **WRITE:** file writes, deletes, subprocesses, HTTP POST/PUT/PATCH/DELETE, SQL INSERT/UPDATE/DELETE, ORM commits,
  sending messages, cloud writes, MCP `readOnlyHint: false`.
- **READ:** file reads, directory listing, HTTP GET, SQL SELECT, or a body with no I/O at all.
- **UNCLEAR:** the body calls something agenticreq cannot follow. Results that depend on it go to Check with a mock run.

Declare what a tool does when its body cannot be followed:

```yaml
tool_effects: {read_file: read, save_report: write}
```

If the body contradicts the declaration, the report shows a gap with the line that writes (AS-232).

## `allow_write:` in mocks.yml

```yaml
allow_write: [workspace/]   # paths relative to the project folder
```

A mock run blocks writes outside the listed paths (and outside `.agenticreq/` and the system temp folder). Allowed
writes are listed in the summary; a blocked write names the file and how to allow it. `agenticreq mocks` pre-fills
folders the code spells out; a folder built from settings must be added by hand.

## When a mock run fails

Every failed or empty run exits 1 with one sentence: what was captured, where it stopped (file:line), why, and the
next step. Calls captured before a failure are kept in `.agenticreq/captures.json`. Typical causes: a wrong `entry:`
in mocks.yml, a dependency missing from the agent's Python, an error in the agent before the model call, an HTTP
client the run cannot capture (only httpx and httpx2 are), a timeout (`--timeout`, default 120 s), or a function that
never calls a model.

## After a mock run

A line stays under Check with a mock run only when editing mocks.yml could settle it (a tool to add to `order:`,
`args:` that reach an agent), and it says what to change. Otherwise it moves: a number in spec.yml with no limit in
code becomes a gap; something that cannot be read from code becomes a sign-off. Running twice with nothing changed
gives the same report.

## Sub-fields only a person can confirm

`role_boundary.capabilities`, `role_boundary.data_access`, `untrusted_inputs[].kind`, `output_size_limits.max_items`,
`write_safety.compensating_action` and `agent_registry[].role` cannot be read from code. Each one a spec writes
becomes a plain question under Human sign-off. `agenticreq check --coverage` lists every parameter.
