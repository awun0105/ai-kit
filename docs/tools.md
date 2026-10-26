# Tools: built-in, MCP and A2A

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
A2A_SERVER_TOKEN=...                               # required; the endpoint runs the agent with its tools
A2A_PUBLIC_URL=https://agent.example.com/a2a       # advertised in the agent card
A2A_AGENT_DESCRIPTION="Answers HR policy questions."
```

- A2A protocol 1.0, JSON-RPC binding, official `a2a-sdk`.
- Agent card (public): `GET /a2a/.well-known/agent-card.json`. JSON-RPC: `POST /a2a/` with
  `Authorization: Bearer <A2A_SERVER_TOKEN>`.
- The A2A `context_id` is the LangGraph thread, so remote callers get multi-turn memory.
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

## Limits

- MCP tools are discovered once at startup; restart to pick up new server tools.
- The A2A server answers synchronously (no streaming, no push notifications) and keeps
  tasks in memory; use `a2a-sdk`'s SQL task store for multi-instance deployments.

## Provenance

| Part | Source |
|---|---|
| MCP over streamable HTTP with bearer auth, evidence tools | Author's solution to the AI20K D09 MultiAgent-MCP-A2A lab (`src/student_agent/mcp_gateway.py`) |
| MCP client | [`langchain.mcp`](https://github.com/langchain-ai/langchain/tree/master/libs/langchain_v1/langchain/mcp) (MIT), [FastMCP](https://github.com/PrefectHQ/fastmcp) (Apache-2.0), [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) (MIT) |
| A2A | [a2a-python](https://github.com/a2aproject/a2a-python) (Apache-2.0), structure from the official [helloworld sample](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents/helloworld) |
