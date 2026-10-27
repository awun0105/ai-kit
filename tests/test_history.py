"""History sent to the model: trimming to the token budget never drops the current turn."""

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from app.utils.graph import prepare_messages

RAG_RESULT = "[1] Điều 5. Người lao động được nghỉ phép năm 12 ngày làm việc. " * 200  # ~3k tokens


def _rag_turn(question: str) -> list:
    call = AIMessage(content="", tool_calls=[{"id": "k1", "name": "knowledge_search", "args": {"query": question}}])
    return [HumanMessage(content=question), call, ToolMessage(content=RAG_RESULT, tool_call_id="k1")]


def _types(messages) -> list[str]:
    return [m.type if hasattr(m, "type") else m.role for m in messages]


def test_a_turn_larger_than_the_budget_is_kept_whole():
    history = [HumanMessage(content="Tên tôi là Lan"), AIMessage(content="Chào Lan")]
    messages = prepare_messages(history + _rag_turn("Tôi được nghỉ phép bao nhiêu ngày?"), "system", max_tokens=2000)

    # Before the fix trim_messages returned [] and the model saw only the system prompt.
    assert _types(messages) == ["system", "human", "ai", "tool"]
    assert messages[1].content == "Tôi được nghỉ phép bao nhiêu ngày?"


def test_older_turns_are_dropped_first():
    old = [HumanMessage(content="câu cũ " * 400), AIMessage(content="trả lời cũ " * 400)]
    messages = prepare_messages(old + [HumanMessage(content="câu mới")], "system", max_tokens=200)

    assert _types(messages) == ["system", "human"]
    assert messages[1].content == "câu mới"


def test_history_within_budget_is_kept():
    history = [HumanMessage(content="a"), AIMessage(content="b"), HumanMessage(content="c")]

    assert _types(prepare_messages(history, "system", max_tokens=2000)) == ["system", "human", "ai", "human"]
