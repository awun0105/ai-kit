"""History compaction: where the cut goes, when a summary is worth paying for, and the graph wiring."""

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver

from app.core import kit
from app.core.config import settings
from app.core.kit import (
    build_history_compactor_from_settings,
    set_history_compactor,
)
from app.core.langgraph.graph import LangGraphAgent
from app.services.compaction import (
    HistoryCompactor,
    HistorySummary,
)

WORDS = "lorem ipsum dolor sit amet " * 8  # ~50 tokens with approx_tokens


class FakeSummarizer:
    def __init__(self, reply: str = "SUMMARY", error: Exception | None = None):
        self.reply, self.error = reply, error
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, previous: str, transcript: str) -> str:
        self.calls.append((previous, transcript))
        if self.error:
            raise self.error
        return self.reply


def _turn(n: int, tool: bool = False, answered: bool = True) -> list:
    messages: list = [HumanMessage(content=f"q{n} {WORDS}", id=f"h{n}")]
    if tool:
        messages.append(
            AIMessage(content="", tool_calls=[{"id": f"c{n}", "name": "search", "args": {}}], id=f"call{n}")
        )
        if answered:
            messages.append(ToolMessage(content=f"result{n} {WORDS}", tool_call_id=f"c{n}", id=f"t{n}"))
    messages.append(AIMessage(content=f"a{n} {WORDS}", id=f"a{n}"))
    return messages


def _compactor(summarizer, trigger=300, keep=150) -> HistoryCompactor:
    return HistoryCompactor(summarizer, trigger_tokens=trigger, keep_tokens=keep)


async def test_short_history_is_left_alone():
    summarizer = FakeSummarizer()

    assert await _compactor(summarizer).compact(_turn(1) + _turn(2)) is None
    assert summarizer.calls == []


async def test_oldest_turns_are_summarized_up_to_a_user_turn():
    summarizer = FakeSummarizer()
    messages = _turn(1, tool=True) + _turn(2) + _turn(3) + _turn(4) + _turn(5)

    summary = await _compactor(summarizer).compact(messages)

    assert summary is not None
    kept = HistoryCompactor.unsummarized(messages, summary)
    assert isinstance(kept[0], HumanMessage)  # never starts on a tool result or an answer
    assert kept[-1].id == "a5"
    assert summary.through_id == messages[len(messages) - len(kept) - 1].id
    assert "TOOL" in summarizer.calls[0][1] and "q1" in summarizer.calls[0][1]
    assert "q5" not in summarizer.calls[0][1]


async def test_a_call_that_is_never_answered_does_not_block_compaction():
    summarizer = FakeSummarizer()
    messages = _turn(1, tool=True, answered=False) + _turn(2) + _turn(3) + _turn(4) + _turn(5)

    summary = await _compactor(summarizer).compact(messages)

    assert summary is not None and summary.through_id != "a1"  # cut past the dead call, not stuck before it


async def test_the_summary_rolls_forward():
    summarizer = FakeSummarizer(reply="NEW")
    messages = _turn(1) + _turn(2) + _turn(3) + _turn(4) + _turn(5) + _turn(6) + _turn(7)
    previous = HistorySummary(text="OLD", through_id="a2")

    summary = await _compactor(summarizer).compact(messages, previous)

    assert summary is not None and summary.text == "NEW"
    assert summarizer.calls[0][0] == "OLD"
    assert "q2" not in summarizer.calls[0][1] and "q3" in summarizer.calls[0][1]


async def test_no_summary_when_it_would_not_save_anything():
    summarizer = FakeSummarizer()
    messages = _turn(1) + _turn(2) + _turn(3) + _turn(4)
    long_previous = HistorySummary(text=WORDS * 10, through_id="a1")

    assert await _compactor(summarizer, trigger=150, keep=100).compact(messages, long_previous) is None
    assert summarizer.calls == []


async def test_a_failed_summary_keeps_the_history():
    compactor = _compactor(FakeSummarizer(error=RuntimeError("provider down")))

    assert await compactor.compact(_turn(1) + _turn(2) + _turn(3) + _turn(4) + _turn(5)) is None


def test_keep_must_be_below_trigger():
    with pytest.raises(ValueError):
        HistoryCompactor(FakeSummarizer(), trigger_tokens=100, keep_tokens=100)


async def test_the_graph_sends_the_summary_instead_of_the_old_turns(monkeypatch):
    seen: list[list[dict]] = []

    async def scripted(messages, *args, **kwargs):
        seen.append(messages)
        return AIMessage(content=f"answer {WORDS}")

    summarizer = FakeSummarizer(reply="The user is Lan, order 123.")
    set_history_compactor(_compactor(summarizer, trigger=300, keep=150))
    agent = LangGraphAgent()
    agent.tools_by_name = {}
    monkeypatch.setattr(agent.llm_service, "call", scripted)
    graph = await agent.create_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "long"}}
    try:
        for n in range(6):
            await graph.ainvoke({"messages": [HumanMessage(content=f"q{n} {WORDS}")]}, config)
    finally:
        set_history_compactor(None)
    state = (await graph.aget_state(config)).values

    assert state["summary"] == "The user is Lan, order 123."
    assert len(state["messages"]) == 12  # the checkpoint keeps every message for the history API
    last_prompt = seen[-1]
    assert "The user is Lan, order 123." in last_prompt[0]["content"]
    assert not any("q0 " in str(m.get("content")) for m in last_prompt[1:])
    assert "q5 " in str(last_prompt[-1]["content"])


async def test_the_settings_summarizer_is_hidden_from_the_stream(monkeypatch):
    configs = []

    async def fake_call(messages, model_name=None, response_format=None, config=None, **kwargs):
        configs.append(config)
        return AIMessage(content="summary")

    monkeypatch.setattr(settings, "HISTORY_COMPACTION_ENABLED", True)
    monkeypatch.setattr(kit.llm_service, "call", fake_call)
    compactor = build_history_compactor_from_settings()
    assert compactor is not None

    assert await compactor._summarize("", "USER: hi") == "summary"
    assert configs == [{"tags": ["nostream"]}]


def test_trigger_above_the_history_budget_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "HISTORY_COMPACTION_ENABLED", True)
    monkeypatch.setattr(settings, "HISTORY_COMPACTION_TRIGGER_TOKENS", settings.HISTORY_MAX_TOKENS + 1)

    with pytest.raises(ValueError):
        build_history_compactor_from_settings()
