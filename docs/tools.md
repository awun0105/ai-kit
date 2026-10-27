# Tools: built-in, MCP, A2A and ACP

The agent's tools come from three places. All of them run through the same tool
node, so the [guardrails](guardrails.md) apply equally: approval for listed tools,
the egress allowlist, and screening of tool output as untrusted data.

| Source | How it gets in | Enabled by |
|---|---|---|
| Built-in | `app/core/langgraph/tools/__init__.py` | always (`knowledge_search` when `RAG_ENABLED`) |
| MCP servers | discovered at startup from `MCP_CONFIG_FILE` | `MCP_CONFIG_FILE` |
| Remote A2A agents | one `ask_remote_agent` tool | `A2A_REMOTE_AGENTS` |

Adding a built-in tool: write a `@tool` function, append it to `tools` in
`app/core/langgraph/tools/__init__.py`, and if it has side effects add its name to
`GUARD_APPROVAL_TOOLS`. Return `(content, artifact)` with
`@tool(response_format="content_and_artifact")` to keep structured data next to the
text the model sees (`knowledge_search` returns its source document ids this way).

## MCP client

```jsonc
// mcp.json — the same "mcpServers" format as Claude Desktop, Cursor and FastMCP
{
  "mcpServers": {
    "orders": {
      "url": "https://orders.internal.example/mcp",
      "headers": { "Authorization": "Bearer ${ORDERS_MCP_TOKEN}" }   // ${VAR} comes from the environment
    },
    "files": { "command": "uvx", "args": ["mcp-server-filesystem", "/srv/docs"] }
  }
}
```

```bash
MCP_CONFIG_FILE=mcp.json
MCP_TOOL_ALLOWLIST=orders_get_order,orders_list_orders   # expose only what the app needs
GUARD_APPROVAL_TOOLS=orders_cancel_order                 # and gate the destructive ones
```

- Built on `langchain.mcp.MCPAdapter` (FastMCP underneath). `langchain-mcp-adapters` is
  archived; its maintained successor is `langchain[mcp]`.
- With several servers, tool names are prefixed with the server name.
- A server that asks the user for input mid-call (MCP elicitation) pauses the graph
  with `interrupt()`; the user's next message answers it.
- A server that fails at startup is logged and skipped; the API still starts.
- Instruction-like text in an MCP result is neutralized and the result's structured
  artifact is dropped, so no unscreened copy is kept.

## The kit as an MCP server

`app/mcp_server.py` exposes `knowledge_search` (read-only) so other agents can use the
same knowledge base. Results pass the output rail, so PII/secrets are redacted there too.

```bash
# stdio, for a local MCP client config
APP_ENV=development uv run python -m app.mcp_server

# streamable HTTP; refuses to start without MCP_SERVER_TOKEN
MCP_SERVER_TOKEN=... APP_ENV=development uv run python -m app.mcp_server --http --port 8765
```

## A2A server

Install the extra (`uv sync --extra a2a`) and set:

```bash
A2A_ENABLED=true
A2A_SERVER_TOKENS=billing=tok-1,crm=tok-2         # one token per calling agent (or A2A_SERVER_TOKEN for one)
A2A_PUBLIC_URL=https://agent.example.com/a2a       # advertised in the agent card
A2A_AGENT_DESCRIPTION="Answers HR policy questions."
```

- A2A protocol 1.0, JSON-RPC binding, official `a2a-sdk`.
- Agent card (public): `GET /a2a/.well-known/agent-card.json`. JSON-RPC: `POST /a2a/` with
  `Authorization: Bearer <token>`. At least one of `A2A_SERVER_TOKEN` (the caller `default`) or
  `A2A_SERVER_TOKENS` is required: the endpoint runs the agent with its tools.
- Each token names a caller. The thread id is `a2a-<caller>-<context_id>`, so a caller gets
  multi-turn memory but can never continue another caller's conversation by reusing its
  context id (the remote side chooses it). The user id is `a2a:<caller>`: long-term memory and
  the cost budget are per caller.
- When a tool needs approval the task ends in `input-required` with the approval question;
  answering in the same context resumes it.

## A2A client

```bash
A2A_REMOTE_AGENTS=billing=https://billing.example.com/a2a,travel=https://travel.example.com/a2a
A2A_REMOTE_TOKEN_BILLING=...                       # per-agent bearer token
```

The model gets one `ask_remote_agent(agent, message)` tool listing the configured names.
It can only reach agents the operator configured, never a URL from a prompt. Replies are
screened like any tool output.

## ACP agent

[Agent Client Protocol](https://agentclientprotocol.com) clients are the apps a user works in:
Zed, JetBrains IDEs, fframes Studio, or your own app in C++, Rust or anything else that can start
a subprocess. They start the agent and talk JSON-RPC over its stdin and stdout. The kit's agent is
one of them:

```bash
uv sync --extra acp
python -m app.acp_agent          # what the client runs; with the offline settings it needs no server
```

In Zed, for example, add it under `agent_servers` in `settings.json` (check
[Zed's external agents docs](https://zed.dev/docs/ai/external-agents) for the current format):

```json
{"agent_servers": {"ai-kit": {"command": "uv",
  "args": ["run", "--directory", "/path/to/your-app", "--extra", "acp", "python", "-m", "app.acp_agent"],
  "env": {"APP_ENV": "development"}}}}
```

- **Sessions.** Each ACP session is one LangGraph thread (`acp-<session id>`); RAG, guardrails,
  retries, the cost guard and history compaction work as in the HTTP app.
- **Approvals.** A tool listed in `GUARD_APPROVAL_TOOLS` becomes an ACP permission request; the
  client's user picks Allow or Reject, and the agent resumes with that answer. A cancelled request
  counts as Reject.
- **Questions.** `ask_human` ends the turn with the question; the user's next prompt answers it.
- **stdout.** stdout carries the protocol, so the entry point imports
  `app.services.acp.reserve_stdout` before anything else: the protocol gets a private copy of
  stdout and file descriptor 1 points at stderr, where every log line and library warning goes.
- **Your own agent.** `KitAcpAgent` serves any `AgentBackend` (`turn(session_id, text) ->
  TurnResult`); `app/acp_agent.py` is the adapter for this app's `LangGraphAgent`.

## Limits

- MCP tools are discovered once at startup; restart to pick up new server tools.
- The A2A server answers synchronously (no streaming, no push notifications) and keeps
  tasks in memory; use `a2a-sdk`'s SQL task store for multi-instance deployments.
- The ACP agent sends each reply as one message (no token streaming), does not load earlier
  sessions (`loadSession` is off), ignores MCP servers the client offers, and stops a cancelled
  turn at its next step rather than in the middle of a model call.

## When you plug it in

- MCP and A2A tools are untrusted: keep them behind the graph's tool node, or wrap them with
  `guard_tools` before handing them to a sub-agent.
- Give each calling agent its own A2A token (`A2A_SERVER_TOKENS`): the caller is part of the
  thread id and the user id, so callers cannot reach each other's conversations and spend is
  counted per caller.
- Limit MCP servers with `MCP_TOOL_ALLOWLIST`; a server may expose far more than you need.
- An ACP entry point must keep stdout for the protocol (import `app.services.acp.reserve_stdout`
  first). One stray `print` on stdout breaks the client.
- The ACP agent runs with the permissions of the user who started the client; the client is the
  trust boundary, so it takes no token. Long-term memory and the cost budget use the user id
  `acp:local`.

## Copy this module

Copy `app/services/mcp/` (and `app/mcp_server.py`) with `tests/test_mcp.py`, or
`app/services/a2a/` with `tests/test_a2a.py`. Neither imports app settings;
`a2a/server.py` uses `app.schemas.Message` in `agent_ask`, the adapter for this app's agent.
For ACP, copy `app/services/acp/` with `tests/test_acp.py`; it needs only `agent-client-protocol`
and `structlog`, and `app/acp_agent.py` shows the adapter to write for your agent.

## Provenance

| Part | Source |
|---|---|
| MCP over streamable HTTP with bearer auth, evidence tools | Author's solution to the AI20K D09 MultiAgent-MCP-A2A lab (`src/student_agent/mcp_gateway.py`) |
| MCP client | [`langchain.mcp`](https://github.com/langchain-ai/langchain/tree/master/libs/langchain_v1/langchain/mcp) (MIT), [FastMCP](https://github.com/PrefectHQ/fastmcp) (Apache-2.0), [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) (MIT) |
| A2A | [a2a-python](https://github.com/a2aproject/a2a-python) (Apache-2.0), structure from the official [helloworld sample](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents/helloworld) |
| ACP | [agentclientprotocol/python-sdk](https://github.com/agentclientprotocol/python-sdk) 0.12.1 (Apache-2.0). Approval and question handling follows fframes Studio's driver model (`desktop/crates/studio-harness/src/driver/api.rs`, commit `a8dc7985`), rewritten |
