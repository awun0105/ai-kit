# In-app agent harness

An agent that works **inside an app**: it reads the app's state, changes it through the app's
own operations, and the user reviews the result before anything is applied. The app may be
written in any language; it only writes a thin **adapter**. The harness runs the flow:

```
task.start ─► draft from the current revision ─► agent turns (tools, permissions)
          ─► validate ─► (one repair turn if it fails) ─► candidate ─► accept ─► revision
                                                                   └─► reject / conflict
```

- Every write lands in a **draft**, never in the real data, until the user accepts.
- `dangerous` operations (send, publish, delete outside the app) always ask the user, and run
  only **after** the revision is committed. A rejected candidate drops them.
- The agent's tools belong to one task, through a **capability** token that dies with the task.
- External agents (Claude Code, Codex, Gemini CLI over ACP) work in a **working directory**
  made from the draft; their file requests outside it are refused.
- Undo applies to the latest revision and goes through validation like any draft.

The design is in [the spec](superpowers/specs/2026-10-27-in-app-agent-harness-design.md); the
wire types are in [`harness/protocol.schema.json`](harness/protocol.schema.json).

## Parts

| Part | Where |
|---|---|
| Harness core (no LangChain): RPC, states, permissions, capabilities, broker, bridge, transcript | `app/services/harness/` |
| ACP driver for external agents (`acp` extra) | `app/services/harness/acp_driver.py` |
| Kit agent driver (LangChain `create_agent` + `guard_tools`) | `app/harness_kit_driver.py` |
| Sidecars | `python -m app.harness_sidecar` (external ACP agent), `python -m app.harness_kit_sidecar` (kit agent) |
| Sample app with a full adapter | `examples/harness_notes/` |
| Conformance suites | `tests/conformance/test_adapter.py`, `tests/conformance/test_harness.py` |
| In-app eval | `app/services/harness/evaluation.py` |

## The connection

The app starts the harness as a child process and both talk **JSON-RPC 2.0 over its stdio**,
one JSON object per line, requests in both directions. The harness logs to stderr only.

**The harness calls the app (the adapter):**

| Method | Params → result |
|---|---|
| `app.describe` | → `AppDescription` (protocol version, operations with JSON schema and level, `supports`) |
| `app.context` | → `AppContext` (revision, a size-bounded view of the state, selection) |
| `draft.open` | `base_revision` → `{draft_id}` |
| `op.call` | `draft_id`, `op`, `args` → any JSON. Reads may run without a draft; writes need one |
| `draft.validate` | `draft_id` → `ValidationReport` |
| `draft.diff` | `draft_id` → `DraftDiff` (`summary` text plus app-defined `detail`) |
| `draft.commit` | `draft_id`, `base_revision` → `CommitResult` (`revision`, or `conflict: true`) |
| `draft.discard` | `draft_id` |
| `revision.undo` | `revision` → `{draft_id}` of a draft that reverses it (only the latest) |
| `draft.materialize` | `draft_id`, `path`: write the draft as files into `path`, replacing what is there (`supports: materialize`) |
| `draft.sync` | `draft_id` → `ValidationReport`: read the folder back into the draft; a malformed file is an issue |
| `effect.run` | `revision`, `effect` (`id`, `op`, `args`): run an approved dangerous operation (`supports: effects`) |

Expected failures are JSON-RPC errors with code `-32000` and a message for the harness.

**The app calls the harness:**

| Method | Params → result |
|---|---|
| `task.start` | `goal`, optional `scope`, `policy` (`ask`, `allow_in_draft`, `deny_writes`), `review` (`auto`, `manual`) → `{task_id}`. Refused while another task is writing |
| `task.cancel` | `task_id` |
| `task.state` | `task_id` → `{state}` |
| `permission.reply` | `id`, `choice` (`allow`, `reject`). A second, late or unknown reply is an error |
| `candidate.accept` | `task_id` → `{revision}`; an error when the app changed since the task started (state `conflict`) |
| `candidate.reject` | `task_id` |
| `revision.undo` | `revision` → `{revision}` |

**Events (notifications to the app):** `task.started`, `task.state {task_id, state}`,
`task.candidate {task_id, diff, effects}`, `task.validation` (the last report of a failed task),
`task.finished {task_id, state, revision}`, `permission.requested {id, task_id, title, kind,
detail}`, `permission.closed {id, outcome}`, `message.delta`, `message.final`, `tool.call`,
`effect.done`, `effect.failed`, `revision.undone`. Every string in an event and in the
transcript has been through the guardrails' secret and PII redaction.

**Sidecar flags** (a port takes the same, so the conformance suite can start it):
`--agent-cmd '<JSON list>'`, `--policy`, `--review`, `--max-repairs N` (default 1),
`--transcripts DIR` (one JSONL file per task).

## Task states

`context_ready → running ⇄ waiting_permission → validating → candidate_ready → accepted`;
`validating → repair_needed → running` once (`--max-repairs`), then `failed`;
`candidate_ready → rejected | conflict`; any state that has not ended → `cancelled` or `failed`.
When a task ends, however it ends, the harness revokes its capability, closes pending
permission requests as `cancelled`, stops the agent, discards the draft (unless accepted) and
deletes the working directory.

## Permissions

| Level | `ask` | `allow_in_draft` (default) | `deny_writes` |
|---|---|---|---|
| `read` | runs | runs | runs |
| `write` (on the draft) | asks | runs | refused |
| `dangerous` | asks, queued | asks, queued | refused |

A permission request the agent makes itself (an ACP agent about to run a shell command) is
asked the same way, with `kind: "agent"`.

## External agents

`python -m app.harness_sidecar --agent-cmd '["npx", "@zed-industries/claude-code-acp"]'`
(needs `uv sync --extra acp`) starts the agent per task, with:

- `cwd` = a temporary folder filled by `draft.materialize`. After every turn the harness calls
  `draft.sync`, then validates.
- The app's operations as an MCP server (`python -m app.services.harness.mcp_bridge`, given in
  the ACP session's `mcp_servers`), relaying to the harness on 127.0.0.1 with the task's
  capability. Before every tool call the harness syncs the folder into the draft, and after a
  write it materializes the draft again, so files and tools see the same draft.
- ACP `fs/read_text_file` and `fs/write_text_file` confined to the folder; the terminal
  capability is not offered.

**Limit:** an agent that runs shell commands with its own tool (not through ACP) can still write
outside the folder. Turn on the agent's own sandbox (for example Codex's workspace-write mode,
Claude Code's permission settings); the harness does not sandbox processes at the OS level.

## When you plug it in

- Write the adapter against the table above and run the adapter suite on it (below) before
  wiring the UI. `examples/harness_notes/notes.py` is a complete one in about 250 lines.
- `app.context` must stay small: send ids and titles, let the agent read details with `read`
  operations.
- Mark every operation that leaves the app (email, upload, payment, delete elsewhere) as
  `dangerous`; the harness only protects what the adapter declares.
- Make `draft.commit` compare the base revision: that is what turns a concurrent change into
  `conflict` instead of a lost update.
- `review: "auto"` commits whatever validates; keep `manual` for anything users care about.
- The kit agent sidecar reads the model and guardrails from the app's settings; add
  `--transcripts` and keep the folder out of backups that leave the machine.

## Copy this module

Copy `app/services/harness/` (and `examples/harness_notes/` as a starting adapter). It needs
`pydantic`, `structlog` and `mcp` (for `mcp_bridge.py`); `acp_driver.py` needs
`agent-client-protocol`; `evaluation.py` needs `app/services/evaluation/trajectory.py`. The
sidecar entry points (`app/harness_cli.py`, `app/harness_sidecar.py`,
`app/harness_kit_sidecar.py`, `app/harness_kit_driver.py`) belong to the app.

## Conformance suites

Both suites talk only through the protocol, so they check code in any language:

```bash
# An app's adapter: a command serving it on stdio ({data} is a temp folder), and its sample operations
HARNESS_ADAPTER_CMD='["./my-app", "--adapter", "--data", "{data}"]' \
HARNESS_ADAPTER_FIXTURE=my_app_fixture.json uv run pytest tests/conformance/test_adapter.py

# A harness port: started with the sidecar flags, against the sample notes app and a scripted ACP agent
HARNESS_CMD='["./my-harness"]' uv run pytest tests/conformance/test_harness.py
```

The fixture names a valid write, a write that fails validation and a dangerous operation
(see `tests/fixtures/harness_notes_fixture.json`). The adapter suite skips the undo,
materialize and effect tests when the adapter does not declare them.

## Eval

`run_in_app_eval(cases, harness_runner(harness), runs=3)` runs each `InAppCase` (goal, expected
tools with a trajectory match mode, a part of `app.context` the end state must contain) through
the harness with automatic review, and reports a pass rate per case and overall.

## Port guide

**Port** into the app's language when the app must ship without Python, start instantly or stay
small: the RPC peer, the state machine, the permission table, capabilities, the broker with its
effect queue, the working-directory handling, and an ACP driver (ACP has SDKs in Rust,
TypeScript, Kotlin). That is about 1,800 lines of Python here, docstrings included. Run the harness suite against
the port.

**Keep in Python and call over MCP** what has no reason to be rewritten: RAG, LLM guardrails,
the eval layer, the kit's agent itself.

**Run the Python sidecar** when shipping a Python runtime is acceptable. Per-call speed is not a
reason to port: the time goes into model calls, not into the harness. Packaging size, start-up
time (about a second for the sidecar) and memory are.

## Not done yet

- No OS-level sandbox of agent processes, no token streaming from the kit agent (one
  `message.delta` per turn), no partial accept of a candidate, undo only of the latest revision.
- One writing task per harness. The kit agent takes the redaction from `guardrails.detectors`,
  which loads the guardrails package (and LangChain) in the sidecar; the core does not.
- Not tried yet with a real Claude Code or Codex ACP adapter; the suites use a scripted agent.

## Sources

Design reduced from fframes Studio (`/home/lqaq/PROJECT/fframes-desktop-awun0105`, commit
`a8dc7985`): `desktop/crates/studio-harness/src/driver/api.rs` and `events.rs` (driver
interface, events), `desktop/crates/studio-engine/src/agent_tools/broker.rs` (per-task
capabilities), `backend/writer_gate.rs` (one writer), `docs/desktop/architecture.md` sections 4,
5 and 9 (draft, validate, accepted revision). Rewritten, no code copied. ACP through
`agent-client-protocol` (agentclientprotocol/python-sdk), MCP through the `mcp` SDK.
