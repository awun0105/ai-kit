"""Multi-turn evaluation: a scripted simulated user talks to the real graph, offline."""

from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver

from app.core.langgraph.graph import LangGraphAgent
from app.core.langgraph.tools.ask_human import ask_human
from app.services.evaluation import (
    Scenario,
    TaskJudge,
    UserSimulator,
    build_report,
    load_scenarios,
    simulate_conversation,
    to_markdown,
)
from tests.fakes import ScriptedModel

SAMPLE = Path(__file__).resolve().parent.parent / "evals" / "datasets" / "orbittech"

SCENARIO = Scenario(
    id="order",
    persona="A customer who writes short messages.",
    goal="Find out when order 123 arrives.",
    opening="Đơn hàng của tôi khi nào tới?",
    success_criteria="The assistant gives the delivery date of order 123.",
    max_turns=4,
)


def _user(message: str = "", done: bool = False) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": "SimulatedTurn", "args": {"done": done, "message": message}, "id": "u"}]
    )


def _verdict(success: bool) -> AIMessage:
    args = {"success": success, "rationale": "gave the date" if success else "never gave the date"}
    return AIMessage(content="", tool_calls=[{"name": "TaskVerdict", "args": args, "id": "v"}])


async def _graph(monkeypatch, replies: list, tools: dict | None = None):
    agent = LangGraphAgent()
    agent.tools_by_name = tools or {}

    async def scripted(messages, *args, **kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(agent.llm_service, "call", scripted)
    return await agent.create_graph(checkpointer=InMemorySaver())


async def test_follow_up_turns_run_on_one_thread_until_the_user_is_done(monkeypatch):
    graph = await _graph(
        monkeypatch, [AIMessage(content="Mã đơn hàng là gì?"), AIMessage(content="Đơn 123 tới ngày 5/11.")]
    )
    simulator = UserSimulator(ScriptedModel(replies=[_user("Mã 123"), _user(done=True)]))

    result = await simulate_conversation(
        graph, simulator, SCENARIO, TaskJudge(ScriptedModel(replies=[_verdict(True)]))
    )

    assert [t["role"] for t in result.transcript] == ["user", "agent", "user", "agent"]
    assert result.transcript[-1]["content"] == "Đơn 123 tới ngày 5/11."
    assert (result.turns, result.stop_reason, result.success) == (2, "user_done", True)


async def test_a_paused_graph_is_resumed_with_the_simulated_reply(monkeypatch):
    ask = AIMessage(content="", tool_calls=[{"name": "ask_human", "args": {"question": "Mã đơn hàng?"}, "id": "h"}])
    graph = await _graph(monkeypatch, [ask, AIMessage(content="Đơn 123 tới ngày 5/11.")], {"ask_human": ask_human})
    simulator = UserSimulator(ScriptedModel(replies=[_user("Mã 123"), _user(done=True)]))

    result = await simulate_conversation(graph, simulator, SCENARIO)

    assert result.transcript[1] == {"role": "agent", "content": "Mã đơn hàng?"}  # the interrupt's question
    assert result.transcript[-1]["content"] == "Đơn 123 tới ngày 5/11."
    state = await graph.aget_state({"configurable": {"thread_id": result.thread_id}})
    tool_replies = [m.content for m in state.values["messages"] if isinstance(m, ToolMessage)]
    assert tool_replies == ["Mã 123"]


async def test_the_turn_cap_stops_a_conversation_that_never_ends(monkeypatch):
    graph = await _graph(monkeypatch, [AIMessage(content="Bạn cần gì?"), AIMessage(content="Bạn cần gì nữa?")])
    simulator = UserSimulator(ScriptedModel(replies=[_user("Đơn của tôi"), _user("Đơn của tôi")]))

    result = await simulate_conversation(graph, simulator, SCENARIO.model_copy(update={"max_turns": 2}))

    assert (result.turns, result.stop_reason) == (2, "turn_limit")


async def test_failed_conversations_count_against_the_gates(monkeypatch):
    graph = await _graph(monkeypatch, [AIMessage(content="Đơn 123 tới ngày 5/11."), RuntimeError("provider down")])
    ok = await simulate_conversation(
        graph,
        UserSimulator(ScriptedModel(replies=[_user(done=True)])),
        SCENARIO,
        TaskJudge(ScriptedModel(replies=[_verdict(True)])),
    )
    broken = await simulate_conversation(graph, UserSimulator(ScriptedModel(replies=[])), SCENARIO)

    report = build_report("t", {}, [], {}, {"conversations.success_rate": 0.9}, conversations=[ok, broken])

    assert broken.stop_reason == "error" and "provider down" in (broken.error or "")
    assert report.conversations["all"]["success_rate"] == 0.5
    assert report.conversations["all"]["error_rate"] == 0.5
    assert not report.passed
    assert "## Conversations to look at" in to_markdown(report)


def test_sample_scenarios_load():
    scenarios = load_scenarios(SAMPLE / "scenarios.json")

    assert len(scenarios) >= 3
    assert len({s.id for s in scenarios}) == len(scenarios)
