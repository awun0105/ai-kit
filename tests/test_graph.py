"""Agent graph tests with a scripted LLM and an in-memory checkpointer."""

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.core.langgraph.graph import LangGraphAgent
from app.core.langgraph.tools.ask_human import ask_human


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


side_effects: list[str] = []


@tool
def send_email(to: str) -> str:
    """Send an email."""
    side_effects.append(to)
    return "sent"


async def test_ask_human_pauses_the_graph_and_resumes_with_the_answer(monkeypatch):
    agent = LangGraphAgent()
    agent.tools_by_name = {"ask_human": ask_human}
    monkeypatch.setattr(
        agent.llm_service,
        "call",
        ScriptedLLM(
            [
                AIMessage(
                    content="", tool_calls=[{"name": "ask_human", "args": {"question": "Which city?"}, "id": "h1"}]
                ),
                AIMessage(content="Booked in Hanoi."),
            ]
        ),
    )
    graph = await agent.create_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "h"}}

    await graph.ainvoke({"messages": [HumanMessage(content="book a room")]}, config)
    paused = await graph.aget_state(config)
    assert paused.next == ("tool_call",) and paused.tasks[0].interrupts[0].value == "Which city?"

    state = await graph.ainvoke(Command(resume="Hanoi"), config)
    answer = next(m for m in state["messages"] if isinstance(m, ToolMessage) and m.name == "ask_human")
    assert answer.content == "Hanoi"


async def test_a_side_effect_next_to_ask_human_runs_once(monkeypatch):
    side_effects.clear()
    agent = LangGraphAgent()
    agent.tools_by_name = {"ask_human": ask_human, "send_email": send_email}
    monkeypatch.setattr(
        agent.llm_service,
        "call",
        ScriptedLLM(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "send_email", "args": {"to": "a@example.com"}, "id": "e1"},
                        {"name": "ask_human", "args": {"question": "Anything else?"}, "id": "h1"},
                    ],
                ),
                AIMessage(content="Done."),
            ]
        ),
    )
    graph = await agent.create_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "h2"}}

    await graph.ainvoke({"messages": [HumanMessage(content="email a then ask me")]}, config)
    assert side_effects == []  # nothing ran before the human answered
    await graph.ainvoke(Command(resume="no"), config)

    assert side_effects == ["a@example.com"]  # once, not again on resume
