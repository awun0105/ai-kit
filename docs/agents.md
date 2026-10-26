# Agents: resilient chat model and the deep-research team

## `ResilientChatModel`

A LangChain `BaseChatModel` whose every call goes through the [reliability](reliability.md)
ladder (failover across registry models, circuit breakers, retry budget). Use it wherever a
library expects a chat model, so new agents inherit the same behaviour as the main one:

```python
from app.services.agents import ResilientChatModel
from app.services.llm import llm_service

model = ResilientChatModel(service=llm_service)                 # starts at DEFAULT_LLM_MODEL
cheap = ResilientChatModel(service=llm_service, start_model="gpt-5.4-nano")

from langchain.agents import create_agent
agent = create_agent(model, tools=[...])                          # tool calling: bind_tools on every rung
verdict = await model.with_structured_output(Verdict).ainvoke(...)  # structured output
```

It is async-only (`ainvoke`, `astream`); the sync API raises `NotImplementedError`.

## Deep research

A lead agent plans the topic (`write_todos`), delegates sub-questions to `researcher`
subagents with the `task` tool, writes `/report.md` with inline `[n]` citations and a
`## Sources` list, and runs `check_report_citations` until it passes. Built on
[`deepagents`](https://github.com/langchain-ai/deepagents); install with `uv sync --extra deep`.

```bash
# as a tool of the chat agent (it receives every other registered tool as a research source)
DEEP_RESEARCH_ENABLED=true
GUARD_APPROVAL_TOOLS=deep_research        # recommended: each run costs tens of model calls

# from the command line
APP_ENV=development uv run python -m scripts.research "survey of world models" > report.md
```

| Variable | Default | Meaning |
|---|---|---|
| `DEEP_RESEARCH_ENABLED` | `false` | Register the `deep_research` tool |
| `DEEP_RESEARCH_MAX_MODEL_CALLS` | `40` | Model calls per run, for the lead and for each researcher |
| `DEEP_RESEARCH_MAX_TOOL_CALLS` | `80` | Tool calls per run, likewise |

Researchers can use any tool the chat agent has: `knowledge_search`, web search, MCP tools,
`ask_remote_agent`. Their findings pass through the same tool node, so tool output is screened.

## Citation checks

`app/services/rag/citations.py` checks citations mechanically:

- `check_answer(answer, source_count)`: every `[n]` must refer to a retrieved source. The chat
  agent runs it on every knowledge answer and counts problems in
  `rag_citation_issues_total{kind="out_of_range"|"uncited"}`.
- `check_report(markdown)`: every `[n]` must have a `## Sources` entry and every entry must be
  cited. The research lead must get "OK" from it before finishing.

Whether a cited source actually supports the sentence is a semantic question for the
[evaluation](evaluation.md) layer.

## Choosing a pattern

| Need | Use |
|---|---|
| Chat with tools, approvals, memory | The main LangGraph agent (`app/core/langgraph/graph.py`) |
| A long, multi-source report | `deep_research` |
| A specialised agent owned by another team | Call it over A2A with `ask_remote_agent` ([tools](tools.md)) |
| A new single-purpose agent inside this app | `create_agent(ResilientChatModel(...), tools)` |

`langgraph-supervisor` and `open_deep_research` are archived upstream; `deepagents` is the
maintained harness for supervisor/sub-agent patterns.

## Provenance

| Part | Source |
|---|---|
| Lead / researcher / citation-validation workflow, call limits | AI20K D23 Deep Agents lab (`agents.py`, `GUIDE.md` part 2; the lab ships TODO stubs, prompts here are new) |
| Citation checking as its own layer | AI20K D16 Agent Arena (`harness/layers/citation_checker.py`) |
| Harness | [deepagents](https://github.com/langchain-ai/deepagents) (MIT), LangChain agent middleware (`TodoListMiddleware`, `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`) |
