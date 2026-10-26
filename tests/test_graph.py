"""Agent graph tests with a scripted LLM and an in-memory checkpointer."""

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from app.core.langgraph.graph import LangGraphAgent


@tool
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


class ScriptedLLM:
    """Stands in for ``LLMService.call`` and returns pre-baked replies in order."""

    def __init__(self, replies: list[AIMessage]):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    async def __call__(self, messages, *args, **kwargs):
        self.calls.append(messages)
        return self.replies.pop(0)


async def _run(agent: LangGraphAgent, text: str) -> list:
    graph = await agent.create_graph(checkpointer=InMemorySaver())
    state = await graph.ainvoke(
        {"messages": [HumanMessage(content=text)]},
        config={"configurable": {"thread_id": "t1"}},
    )
    return state["messages"]


async def test_plain_answer_ends_after_one_llm_call(monkeypatch):
    agent = LangGraphAgent()
    llm = ScriptedLLM([AIMessage(content="hello there")])
    monkeypatch.setattr(agent.llm_service, "call", llm)

    messages = await _run(agent, "hi")

    assert messages[-1].content == "hello there"
    assert len(llm.calls) == 1


async def test_tool_call_round_trip(monkeypatch):
    agent = LangGraphAgent()
    agent.tools_by_name = {"add": add}
    llm = ScriptedLLM(
        [
            AIMessage(content="", tool_calls=[{"name": "add", "args": {"a": 2, "b": 3}, "id": "call-1"}]),
            AIMessage(content="2 + 3 = 5"),
        ]
    )
    monkeypatch.setattr(agent.llm_service, "call", llm)

    messages = await _run(agent, "what is 2 + 3?")

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert [m.content for m in tool_messages] == ["5"]
    assert messages[-1].content == "2 + 3 = 5"
    assert len(llm.calls) == 2
