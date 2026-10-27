# In-app agent harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A language-neutral harness that lets an agent work inside an app through drafts, permissions, validation and revisions, with a Python reference implementation, a sample app and two conformance suites.

**Architecture:** The app and the harness share one bidirectional JSON-RPC 2.0 connection (one JSON object per line): the harness calls the app's adapter (`app.*`, `draft.*`, `op.call`, `effect.run`), the app calls the harness (`task.*`, `permission.reply`, `candidate.*`, `revision.undo`) and receives event notifications. The harness core (`app/services/harness/`) has no LangChain import; agents plug in through an `AgentDriver` (external agents over ACP, the kit's agent in process). External agents reach the task's tools through a stdio MCP bridge that relays to the harness over a local TCP socket, authenticated by the task's capability token.

**Tech Stack:** Python 3.13, asyncio, pydantic v2, `agent-client-protocol` 0.12.1 (extra `acp`), `mcp` 2.3 low-level `Server` / `ClientSession` (already installed through `langchain[mcp]`), LangChain `create_agent` for the kit driver only, pytest.

**Spec:** `docs/superpowers/specs/2026-10-27-in-app-agent-harness-design.md`

## Global Constraints

- Toolbox rule: `app/services/harness/*` never imports `app.core.config`, `app.core.logging`, `app.services.llm`, `app.services.database`, LangChain or LangGraph. Log with `structlog.get_logger()`. `tests/test_toolbox_imports.py` gains `app.services.harness` and checks no `langchain` module is loaded.
- Modules needing the `acp` extra (`acp_driver.py`) are not imported by `app/services/harness/__init__.py`.
- Tests run offline: scripted ACP agent, `ScriptedModel` from `tests/fakes.py`.
- Operation levels: `read`, `write`, `dangerous`. Policies: `ask`, `allow_in_draft`, `deny_writes`. `dangerous` always asks; under `deny_writes` it is denied.
- Review policies: `auto`, `manual`. At most `max_repairs=1` repair turn by default. One active writing task per harness.
- Approved `dangerous` operations are queued as effects and run, in order, only after `draft.commit` succeeds; dropped on reject, conflict, cancel or failure.
- External agents run with `cwd` = a working directory the harness creates and passes to `draft.materialize(draft_id, path)`; ACP `fs/*` outside it is refused; ACP terminal is not advertised; the directory is deleted when the task ends.
- Capability tokens: issued at task start, TTL configurable (default 3600 s), revoked when the task ends however it ends.
- Undo applies only to the latest revision the harness committed.
- Never return exception text to the app or agent; log it and return a generic message.
- Record the fframes sources (repo path, commit `a8dc7985`) in `docs/harness.md`.

## Review Focus

- An agent tool call that arrives while the task is not running (after cancel, during validation) → refused, no draft change. Test in Task 4.
- A permission reply for a task that was cancelled meanwhile → the pending ask resolves as rejected, the reply returns an error. Test in Task 3.
- `draft.sync` reporting a malformed file → counts as a failed validation (repair turn), not a crash. Test in Task 9 (harness suite).
- The external agent process exits mid-turn → task goes to `failed`, working dir removed, capability revoked. Test in Task 9.
- An effect raising after commit → revision kept, `effect.failed` event, later effects still run. Test in Task 5.

---

### Task 1: JSON-RPC peer and protocol models

**Files:**
- Create: `app/services/harness/__init__.py`, `app/services/harness/rpc.py`, `app/services/harness/protocol.py`
- Test: `tests/test_harness_core.py`

**Interfaces:**
- Produces:
  - `class RpcError(Exception)(code: int, message: str, data: Any = None)`; codes `METHOD_NOT_FOUND=-32601`, `INVALID_PARAMS=-32602`, `INTERNAL=-32603`, `APP_ERROR=-32000`.
  - `class JsonRpcPeer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, handlers: dict[str, Callable[[dict], Awaitable[Any]]] | None = None)`: `start()`, `async request(method, params: dict | None = None, timeout: float | None = None) -> Any`, `async notify(method, params)`, `on(method, handler)`, `async close()`, `closed: asyncio.Event`. Handler exceptions other than `RpcError` become `INTERNAL` with message `"internal error"`.
  - `async def open_stdio_peer(handlers) -> JsonRpcPeer` (this process's stdin + reserved stdout fd passed in), `async def spawn_peer(cmd: list[str], handlers, cwd=None, env=None) -> tuple[JsonRpcPeer, asyncio.subprocess.Process]`.
  - `protocol.py`: `PROTOCOL_VERSION = "0.1"`, `OpLevel`, `Operation(name, description, params: dict, level)`, `AppDescription(protocol_version, operations: list[Operation], supports: list[Literal["undo", "materialize", "effects"]])`, `ValidationIssue(code, message, path: str | None)`, `ValidationReport(ok: bool, issues: list[ValidationIssue])`, `Effect(id, op, args)`, `CommitResult(revision: str | None, conflict: bool)`, `AppContext(revision: str, state: Any, selection: Any = None)`, `def protocol_schema() -> dict`.

- [ ] Test `test_rpc_requests_both_ways_and_notifications`: two peers over `os.pipe` pairs; A calls B's `add` → 3, B calls A's `echo`, B notifies A; unknown method → `RpcError` code -32601; a handler raising `ValueError("secret")` → error message `"internal error"` (no `secret`).
- [ ] Test `test_protocol_schema_is_published`: `protocol_schema()` equals `json.load(docs/harness/protocol.schema.json)`.
- [ ] Implement; write `scripts/harness_schema.py` that writes the schema file; run it.
- [ ] Commit `feat(harness): JSON-RPC peer and protocol models`.

### Task 2: Task state machine, capabilities, transcript

**Files:**
- Create: `app/services/harness/state.py`, `capability.py`, `transcript.py`
- Test: `tests/test_harness_core.py`

**Interfaces:**
- Produces:
  - `class TaskState(StrEnum)`: `context_ready, running, waiting_permission, validating, repair_needed, candidate_ready, accepted, rejected, conflict, failed, cancelled`; `TERMINAL`, `ACTIVE` frozensets; `TRANSITIONS: dict[TaskState, frozenset[TaskState]]` exactly as spec 5.4 (+ every active → `failed`, `cancelled`); `class InvalidTransition(Exception)`; `class TaskStateMachine(on_change: Callable[[TaskState, TaskState], None] | None)` with `.state`, `.to(new)`.
  - `class CapabilityError(Exception)`; `class CapabilityStore(ttl_s: float = 3600, clock=time.monotonic)`: `issue(task_id) -> str`, `resolve(token) -> str`, `revoke(task_id)`.
  - `class Transcript(path: Path, redact: Callable[[str], str] = lambda s: s)`: `write(event: str, data: dict)` appends one JSON line with `ts`, redacting every string value recursively.
- [ ] Tests: `test_state_machine_follows_the_spec` (`running→validating` ok, `candidate_ready→running` raises, terminal states have no exits, cancel from every active state); `test_capability_expires_and_is_revoked` (fake clock past TTL → `CapabilityError`; revoke → `CapabilityError`); `test_transcript_redacts` (redact replacing `sk-` keys).
- [ ] Implement, run, commit `feat(harness): task states, capabilities, transcript`.

### Task 3: Permission desk and tool broker

**Files:**
- Create: `app/services/harness/permissions.py`, `app/services/harness/broker.py`
- Test: `tests/test_harness_core.py`

**Interfaces:**
- Produces:
  - `class PermissionPolicy(StrEnum)`: `ask, allow_in_draft, deny_writes`; `def decide(level, policy) -> Literal["allow", "ask", "deny"]`.
  - `class PermissionDesk(emit: Callable[[str, dict], Awaitable[None]])`: `async ask(task_id, title, kind: str, detail: dict | None = None) -> bool`; `reply(request_id, choice: Literal["allow", "reject"]) -> None` raising `RpcError(APP_ERROR, "unknown or closed permission request")` for unknown, duplicate or late ids; `close_task(task_id)` resolves pending asks as rejected. Emits `permission.requested {id, task_id, title, kind, detail}` and `permission.closed {id, outcome: "allow"|"reject"|"cancelled"}`.
  - `class ToolBroker(lookup: Callable[[str], "TaskRuntime"], caps: CapabilityStore, desk: PermissionDesk)`: `async list_tools(token) -> list[Operation]` (task ops + `task_context` + `validate_draft`); `async call(token, name, args) -> ToolResult(text: str, is_error: bool)`.
  - `TaskRuntime` (dataclass in `broker.py`, filled by Task 5): `id, goal, scope, policy, draft_id, base_revision, operations: dict[str, Operation], effects: list[Effect], machine: TaskStateMachine, adapter: AdapterClient, tools_called: list[str]`.
- [ ] Tests (fake adapter recording calls): `test_read_runs_without_asking`; `test_write_asks_under_ask_and_not_under_allow_in_draft`; `test_dangerous_always_asks_and_is_queued_not_run` (adapter not called, `effects` has it, text contains `queued`); `test_deny_writes_refuses_write_and_dangerous`; `test_rejected_permission_returns_error_result`; `test_duplicate_and_unknown_permission_replies_are_refused`; `test_cancelled_task_resolves_pending_ask_as_rejected`; `test_calls_outside_running_are_refused` (state `validating`); `test_bad_token_is_refused`. Asking moves `running → waiting_permission → running`.
- [ ] Implement, run, commit `feat(harness): permission policy and tool broker`.

### Task 4: Adapter client and Python adapter server

**Files:**
- Create: `app/services/harness/adapter.py`
- Test: covered by Task 6 adapter suite and Task 3 fakes.

**Interfaces:**
- Produces:
  - `class AdapterClient(peer: JsonRpcPeer)`: `describe() -> AppDescription`, `context() -> AppContext`, `open_draft(base_revision) -> str`, `call(draft_id | None, op, args) -> Any`, `validate(draft_id) -> ValidationReport`, `diff(draft_id) -> dict` (must hold `summary: str`), `commit(draft_id, base_revision) -> CommitResult`, `discard(draft_id)`, `undo(revision) -> str` (draft id), `materialize(draft_id, path: str)`, `sync(draft_id) -> ValidationReport`, `run_effect(revision, effect: Effect) -> Any`.
  - `class AppAdapter(Protocol)` with the same methods (sync or async allowed) and `def adapter_handlers(adapter) -> dict[str, Handler]` mapping wire names: `app.describe, app.context, draft.open, op.call, draft.validate, draft.diff, draft.commit, draft.discard, revision.undo, draft.materialize, draft.sync, effect.run`. Wire params use snake_case keys: `base_revision, draft_id, op, args, revision, effect, path`.
- [ ] Implement with Task 6; commit together.

### Task 5: Harness core and sidecar API

**Files:**
- Create: `app/services/harness/driver.py`, `app/services/harness/harness.py`, `app/services/harness/sidecar.py`
- Test: `tests/test_harness_core.py` (fake driver, in-memory notes adapter from Task 6)

**Interfaces:**
- Produces:
  - `driver.py`: `TurnContext` dataclass `(task_id, goal, scope, workdir: Path | None, token, list_tools: Callable[[], Awaitable[list[Operation]]], call_tool: Callable[[str, dict], Awaitable[ToolResult]], ask_permission: Callable[[str, str, dict | None], Awaitable[bool]], emit: Callable[[str, dict], Awaitable[None]], bridge: BridgeInfo | None)`; `class AgentDriver(Protocol)`: `needs_workdir: bool`, `async run_turn(ctx, prompt: str) -> str` (final text), `async cancel()`, `async close()`; `DriverFactory = Callable[[], AgentDriver]`.
  - `class Harness(adapter: AdapterClient, driver_factory, emit, *, policy=PermissionPolicy.allow_in_draft, review: Literal["auto","manual"]="manual", max_repairs=1, capability_ttl_s=3600, redact=identity, transcript_dir: Path | None = None, bridge: ToolBridge | None = None)`: `async start_task(goal, scope=None, policy=None, review=None) -> str` (raises `RpcError` `"another task is writing"`), `async cancel(task_id)`, `reply_permission(id, choice)`, `async accept(task_id) -> str` revision, `async reject(task_id)`, `async undo(revision) -> str`, `async wait(task_id) -> TaskState`, `async close()`.
  - Turn prompts: first turn = goal + scope; repair turn = `"Validation failed. Fix these problems, then finish:\n- " + issues`.
  - Events: `task.state {task_id, state}`, `task.candidate {task_id, diff, effects}`, `message.delta {task_id, text}`, `tool.call {task_id, name, args, level}`, `effect.done {task_id, revision, effect_id}`, `effect.failed {task_id, revision, effect_id}`, `task.finished {task_id, state, revision}` plus permission events.
  - `sidecar.py`: `def harness_handlers(harness) -> dict` for `task.start {goal, scope?, policy?, review?} -> {task_id}`, `task.cancel {task_id}`, `permission.reply {id, choice}`, `candidate.accept {task_id} -> {revision}`, `candidate.reject {task_id}`, `revision.undo {revision} -> {revision}`.
- [ ] Tests: `test_task_runs_to_candidate_without_touching_data` (manual review, context unchanged until accept); `test_auto_review_commits`; `test_one_repair_then_failed`; `test_repair_succeeds`; `test_effects_run_after_commit_in_order`; `test_effect_failure_keeps_revision_and_runs_the_rest`; `test_effects_dropped_on_reject`; `test_conflict_when_base_changed`; `test_single_writer`; `test_cancel_while_running_and_while_waiting_permission`; `test_capability_revoked_after_task`; `test_undo_latest_only`; `test_driver_crash_fails_task_and_cleans_up` (workdir gone).
- [ ] Implement, run, commit `feat(harness): task orchestration and the app-facing API`.

### Task 6: Sample notes app and adapter conformance suite

**Files:**
- Create: `examples/__init__.py`, `examples/harness_notes/__init__.py`, `examples/harness_notes/notes.py`, `examples/harness_notes/__main__.py`
- Create: `tests/conformance/__init__.py`, `tests/conformance/conftest.py`, `tests/conformance/test_adapter.py`
- Create: `tests/fixtures/harness_notes_fixture.json`

**Interfaces:**
- Produces:
  - `class NotesAdapter(data_dir: Path)` implementing `AppAdapter`: store `notes.json` `{revision: int, notes: {id: {title, body}}}`, snapshot per revision in `revisions/<n>.json`; operations `list_notes` (read), `read_note {id}` (read), `write_note {id, title, body}` (write), `delete_note {id}` (write), `publish_note {id}` (dangerous; effect writes `outbox/<id>.md`). Validation: title non-empty, at most 80 characters. `materialize` writes `<id>.md` with first line `# <title>`; `sync` issue `bad_file` for a file not starting with `# `. Supports `undo, materialize, effects`.
  - `python -m examples.harness_notes --data DIR` serves the adapter on stdio (adapter only).
  - Fixture JSON (env `HARNESS_ADAPTER_FIXTURE`): `{"write": {op, args}, "invalid_write": {op, args}, "dangerous": {op, args}}`. Env `HARNESS_ADAPTER_CMD` (JSON list, `{data}` replaced by a temp dir).
- [ ] Adapter suite tests: `test_describe`, `test_write_on_draft_leaves_data_alone`, `test_commit_creates_a_revision`, `test_stale_base_conflicts`, `test_invalid_write_fails_validation`, `test_discard`, `test_write_without_draft_is_refused`, `test_undo_restores_previous` (skip unless `undo`), `test_materialize_and_sync_round_trip` (skip unless `materialize`), `test_effect_runs_after_commit` (skip unless `effects`).
- [ ] Implement Task 4 + 6, run `uv run pytest tests/conformance/test_adapter.py`, commit `feat(harness): sample notes app and adapter conformance suite`.

### Task 7: Tool bridge for external agents

**Files:**
- Create: `app/services/harness/bridge.py`, `app/services/harness/mcp_bridge.py`
- Test: `tests/test_harness_core.py::test_bridge_lists_and_calls_through_mcp`

**Interfaces:**
- Produces:
  - `BridgeInfo(host, port)`; `class ToolBridge(broker: ToolBroker)`: `async start() -> BridgeInfo` (127.0.0.1, port 0; JSON-RPC methods `tools.list {token}`, `tools.call {token, name, args}`), `async close()`; `def mcp_server_spec(info, token) -> dict` = `{"name": "app", "command": sys.executable, "args": ["-m", "app.services.harness.mcp_bridge"], "env": {"HARNESS_BRIDGE": "host:port", "HARNESS_CAPABILITY": token}}`.
  - `mcp_bridge.py`: `python -m app.services.harness.mcp_bridge` runs an MCP stdio server (mcp low-level `Server`) relaying `tools/list` and `tools/call`; errors come back as `is_error` results.
- [ ] Test drives a real MCP `ClientSession` over `stdio_client` against the bridge with a fake broker; revoked token → error result.
- [ ] Implement, run, commit `feat(harness): MCP bridge to the task's tools`.

### Task 8: ACP driver and scripted ACP agent

**Files:**
- Create: `app/services/harness/acp_driver.py`, `tests/conformance/scripted_agent.py`
- Test: Task 9 suite

**Interfaces:**
- Produces:
  - `class AcpDriver(command: list[str], env: dict | None = None)`, `needs_workdir = True`. First `run_turn` spawns the agent (`spawn_agent_process`, `cwd=workdir`), initializes with `fs.read_text_file/write_text_file=True`, `terminal=False`, `new_session(cwd=workdir, mcp_servers=[McpServerStdio from mcp_server_spec])`; each turn `prompt`; `session_update` agent message chunks → `message.delta`, tool calls → `tool.call`; `request_permission` → `ctx.ask_permission(title, "agent", ...)` mapped to the first allow / reject option; `read_text_file`/`write_text_file` resolve the path and refuse anything outside `workdir` with `RequestError`. `cancel()` sends ACP cancel; `close()` ends the process. A process exit mid-turn raises `DriverError`.
  - `scripted_agent.py SCRIPT LOG`: ACP agent; turn *n* runs `script["turns"][n]`, a list of steps `{"tool": name, "args": {}}` (via MCP client on the session's stdio MCP server), `{"write_file": rel_or_abs_path, "content": str}`, `{"ask": title}` (ACP `request_permission`), `{"say": text}`, `{"sleep": seconds}`, `{"exit": code}`. Appends `{"turn", "step", "result"}` lines to LOG and the session's `mcp_servers` on the first turn.
- [ ] Implement with Task 9, commit together.

### Task 9: Sidecar entry point and harness conformance suite

**Files:**
- Create: `app/harness_sidecar.py`, `tests/conformance/test_harness.py`
- Modify: `pyproject.toml` (nothing new: `acp` extra already holds the SDK)

**Interfaces:**
- Produces:
  - CLI contract (spec, for ports): `<harness> --agent-cmd '<JSON list>' [--policy P] [--review auto|manual] [--max-repairs N] [--transcripts DIR]`; the harness speaks the protocol on stdio, logs on stderr. `python -m app.harness_sidecar` also takes `--agent kit` (Task 10).
  - Suite env `HARNESS_CMD` (JSON list, default `[python, -m, app.harness_sidecar]`). The test plays the app: serves `NotesAdapter` on the harness's stdio and calls the app→harness API.
- [ ] Tests: `test_writes_stay_in_draft_until_accept`, `test_dangerous_asks_queues_and_runs_after_accept`, `test_dangerous_dropped_on_reject`, `test_permission_reply_twice_or_unknown_is_refused`, `test_validation_failure_gets_one_repair`, `test_malformed_file_is_a_validation_failure`, `test_conflict_when_base_changed`, `test_undo`, `test_fs_outside_workdir_is_refused`, `test_capability_dead_after_task` (respawn the logged MCP server and call `task_context` → error), `test_cancel_running_and_waiting`, `test_agent_exit_fails_task`, `test_single_writer`, `test_deny_writes`.
- [ ] Implement Tasks 8 + 9, run `uv run pytest tests/conformance -q`, commit `feat(harness): ACP driver, sidecar and harness conformance suite`.

### Task 10: Kit driver and in-app eval

**Files:**
- Create: `app/harness_kit_driver.py`, `app/services/harness/evaluation.py`
- Test: `tests/test_harness_kit.py`

**Interfaces:**
- Produces:
  - `class KitDriver(model: BaseChatModel, guard: Guardrails)`, `needs_workdir = False`; tools built per turn as `StructuredTool(name, description, args_schema=<JSON schema dict>, coroutine=...)` wrapped with `guard_tools`, run with `langchain.agents.create_agent`; message history kept across the task's turns; text deltas not streamed (final text emitted once).
  - `evaluation.py`: `InAppCase(id, goal, expected_tools: list[str], tool_match: ToolMatch = "in_order", expect: dict)`, `CaseRun(state: str, tools_called: list[str], context: Any)`, `async def run_in_app_eval(cases, run_case: Callable[[InAppCase], Awaitable[CaseRun]], runs=1) -> InAppReport(pass_rate: float, items: list[...])`; a run passes when `state == "accepted"`, `match_trajectory(...)`, and every `expect` key/value is contained in `context`.
- [ ] Tests: `test_kit_driver_edits_through_the_harness` (ScriptedModel calls `write_note`, harness auto review, notes changed); `test_in_app_eval_pass_rate` (two runs, one passing → 0.5).
- [ ] Implement, run, commit `feat(harness): kit agent driver and in-app eval`.

### Task 11: Docs, wiring, final checks

**Files:**
- Create: `docs/harness.md` (protocol summary, sidecar CLI, events, When you plug it in, Copy this module, Port guide: what to port vs keep in Python, how to run both suites against a port, sidecar vs port, limits, sources).
- Modify: `docs/ROADMAP.md` (layer 12), `AGENTS.md` (Kit Layers list), `README.md` + `README_EN.md` (one line each), `tests/test_toolbox_imports.py`, spec (`draft.materialize(draft_id, path)`, status line), `~/.agents/skills/ai-kit/SKILL.md`.
- [ ] `uv run pytest -q`, `uv run ruff check . && uv run ruff format --check .`, `uv run pyright`; `graphify update .`.
- [ ] Commit `docs(harness): guide, port notes, roadmap`; push `mine main` (user asked to push); push skill repo.
