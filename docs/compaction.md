# Conversation history

What the model sees of a conversation on each call, and what happens when the conversation
outgrows the budget.

## The budget

`prepare_messages` (`app/utils/graph.py`) sends the system prompt and then as much recent
history as fits in `HISTORY_MAX_TOKENS` (16000). History is cut at user messages, oldest turn
first, so a tool result never loses the call that produced it. The **current turn** (the
question and its tool round trips) is always sent whole, even when it alone is over the budget:
a large RAG result must not leave the model without the question.

`HISTORY_MAX_TOKENS` is separate from `MAX_TOKENS`, the limit on each model reply.

## Compaction (off by default)

Cutting forgets the start of the conversation: the user's name, the order number given ten
turns ago. With `HISTORY_COMPACTION_ENABLED=true` the oldest turns are replaced by a rolling
summary instead:

1. Before each model call, the chat node asks `HistoryCompactor.compact` whether the history
   not yet summarized has reached `HISTORY_COMPACTION_TRIGGER_TOKENS` (12000).
2. If so, it keeps the newest turns that fit `HISTORY_COMPACTION_KEEP_TOKENS` (4000, at least
   the current turn) word for word and summarizes everything before them, merged with the
   previous summary.
3. The summary and the id of the last message it covers go into the graph state (`summary`,
   `summary_through`). Later calls send the system prompt with the summary appended, then the
   messages after that id.

The checkpoint keeps every message, so `GET /chatbot/messages` still returns the whole
conversation; only the prompt changes.

Where the cut goes:

- Only before a user message with no tool call still waiting for its result, so the kept part
  never starts with an orphaned tool result (providers reject those).
- A tool call that is never answered anywhere in the conversation (the process died mid-call)
  is ignored, otherwise it would block every later compaction.
- No summary is paid for when the part it would replace is no larger than the previous summary,
  the best estimate of how long the new one will be.
- If the summarizer fails, the turn goes on with the unsummarized history.

Cost: one extra model call (`HISTORY_COMPACTION_MODEL`, default model when empty) each time it
triggers, roughly every `TRIGGER - KEEP` tokens of new conversation. The call goes through
`llm_service` (retries, fallback, Langfuse, cost guard) and is tagged `nostream`.

| Setting | Default | |
|---|---|---|
| `HISTORY_MAX_TOKENS` | `16000` | history budget per call |
| `HISTORY_COMPACTION_ENABLED` | `false` | |
| `HISTORY_COMPACTION_TRIGGER_TOKENS` | `12000` | must not exceed `HISTORY_MAX_TOKENS` (checked at startup), or turns are cut before they are summarized |
| `HISTORY_COMPACTION_KEEP_TOKENS` | `4000` | must be below the trigger |
| `HISTORY_COMPACTION_MODEL` | `""` | summarizer model; a small one is enough |

## Code

| File | Responsibility |
|---|---|
| `app/utils/graph.py` | `prepare_messages`: budget, cut at user messages, current turn kept |
| `app/services/compaction.py` | `HistoryCompactor`, `HistorySummary`, the cut and worth-it rules |
| `app/core/kit.py` | `build_history_compactor_from_settings`, `get_history_compactor`, `set_history_compactor` |
| `app/core/langgraph/graph.py` | `_chat`: compacts, then sends summary + newer messages |
| `tests/test_history.py`, `tests/test_compaction.py` | Offline tests |

## When you plug it in

- Tag the summarizer's model call `nostream`, or the summary is streamed to the client as part
  of the answer (it runs inside the chat node).
- Keep `TRIGGER <= HISTORY_MAX_TOKENS`.
- The summary holds what the user said, so treat it like the conversation: it is per thread,
  never shared, and the answer cache never sees it (cached turns are first questions).

## Copy this module

Copy `app/services/compaction.py` with `tests/test_compaction.py` (drop the two tests that use
the app's graph and settings). It needs `langchain-core`, `pydantic` and `structlog`. Pass your
own `summarize(previous, transcript)` and a token counter (`approx_tokens` is a rough default).

## Provenance

| Part | Source |
|---|---|
| Cut only at balanced points, dead-call second pass, worth-it check against the previous summary, keep going on summarizer errors | Ideas from Google ADK event compaction (google/adk-python, commit `2084efb`): `src/google/adk/apps/compaction.py` (`_longest_self_contained_prefix`, `_provably_dead_call_ids`, `_is_range_worth_compacting`) and `plugins/context_filter_plugin.py`. Rewritten for LangChain messages and LangGraph state, no code copied |
| Empty-history bug (one large turn trimmed to nothing) | Found while comparing with ADK; `tests/test_history.py` |
