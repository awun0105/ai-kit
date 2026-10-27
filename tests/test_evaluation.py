"""Evaluation layer: metrics, golden-set loading, judge controls, and the offline retrieval gate."""

import math
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.core.kit import set_rag_pipeline
from app.core.langgraph.graph import LangGraphAgent
from app.core.langgraph.tools.knowledge_search import knowledge_search
from app.services.evaluation import (
    AnswerGrade,
    AnswerJudge,
    GoldenItem,
    build_report,
    cohen_kappa,
    evaluate_answers,
    evaluate_retrieval,
    load_golden,
    score_retrieval,
    to_markdown,
)
from app.services.evaluation.runner import (
    AnswerRun,
    ItemResult,
)
from app.services.evaluation.trajectory import match_trajectory
from app.services.rag import (
    HashingEmbedder,
    InMemoryVectorStore,
    RAGPipeline,
)
from app.services.rag.chunking import load_documents
from tests.fakes import ScriptedModel

SAMPLE = Path(__file__).resolve().parent.parent / "evals" / "datasets" / "orbittech"


def test_retrieval_metrics():
    s = score_retrieval(["b.md", "b.md", "a.md", "c.md"], expected=["a.md", "c.md"], k=3)

    assert s.hit == 1.0
    assert s.recall == 1.0  # chunks collapse to documents: a.md and c.md are both in the top 3 documents
    assert s.reciprocal_rank == 0.5
    expected_ndcg = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert s.ndcg == pytest.approx(expected_ndcg)
    assert score_retrieval(["x.md"], ["a.md"], k=5).hit == 0.0


def test_load_d14_golden_set():
    items = load_golden(SAMPLE / "golden.json")

    assert len(items) == 20
    first = next(i for i in items if i.id == "E01")
    assert first.expected_sources == ["06_warranty_policy.md"]
    assert sum(i.should_refuse for i in items) == 3


def test_d14_items_expect_the_retrieval_tool_when_asked():
    items = {i.id: i for i in load_golden(SAMPLE / "golden.json", retrieval_tool="knowledge_search")}

    assert items["E01"].expected_tools == ["knowledge_search"]
    assert items["A01"].expected_tools == []  # adversarial: declining without a search is fine
    assert load_golden(SAMPLE / "golden.json")[0].expected_tools == []


@pytest.mark.parametrize(
    ("actual", "expected", "mode", "matches"),
    [
        (["search", "lookup"], ["search", "lookup"], "exact", True),
        (["search", "search", "lookup"], ["search", "lookup"], "exact", False),
        (["search", "calc", "lookup"], ["search", "lookup"], "in_order", True),
        (["lookup", "search"], ["search", "lookup"], "in_order", False),
        (["lookup", "search"], ["search", "lookup"], "any_order", True),
        (["search"], ["search", "search"], "any_order", False),
        ([], [], "exact", True),
        (["search"], [], "in_order", True),
    ],
)
def test_match_trajectory(actual, expected, mode, matches):
    assert match_trajectory(actual, expected, mode) is matches


def test_load_jsonl(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text('{"id": "1", "question": "q?", "expected_sources": ["a.md"]}\n', encoding="utf-8")
    assert load_golden(path)[0].expected_sources == ["a.md"]


def test_cohen_kappa():
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [1, 1, 0, 0]) == 0.0
    assert cohen_kappa(["pass", "pass", "fail", "pass"], ["pass", "fail", "fail", "pass"]) == pytest.approx(0.5)


async def test_offline_retrieval_regression_gate():
    """The CI gate: hybrid retrieval on the sample corpus must not regress (hashing embedder, so BM25 carries it)."""
    pipeline = RAGPipeline(HashingEmbedder(1024), InMemoryVectorStore())
    await pipeline.ingest(load_documents(SAMPLE / "corpus"))
    items = load_golden(SAMPLE / "golden.json")

    results = await evaluate_retrieval(pipeline, items, k=5)
    report = build_report(
        "orbittech",
        {},
        items,
        results,
        {"retrieval.hit": 0.9, "retrieval.recall": 0.85, "retrieval.reciprocal_rank": 0.75},
    )

    assert report.passed, to_markdown(report)
    assert report.retrieval["easy"]["hit"] == 1.0


def _grade_call(**scores):
    args = {"faithfulness": 5, "correctness": 5, "citation_support": 5, "declined": False, "rationale": "ok"} | scores
    return AIMessage(content="", tool_calls=[{"name": "AnswerGrade", "args": args, "id": "g"}])


async def test_answers_are_run_through_the_real_graph_and_judged(monkeypatch):
    pipeline = RAGPipeline(HashingEmbedder(1024), InMemoryVectorStore())
    await pipeline.ingest(load_documents(SAMPLE / "corpus"))
    set_rag_pipeline(pipeline)
    items = [
        GoldenItem(
            id="w",
            question="What is the warranty for the NovaBook 14?",
            expected_answer="24 months",
            difficulty="easy",
            expected_tools=["knowledge_search"],
        ),
        GoldenItem(
            id="adv",
            question="Why is everything free forever?",
            expected_answer="It is not.",
            difficulty="adversarial",
            should_refuse=True,
        ),
    ]
    agent_replies = [
        AIMessage(
            content="", tool_calls=[{"name": "knowledge_search", "args": {"query": "NovaBook warranty"}, "id": "k"}]
        ),
        AIMessage(content="The NovaBook 14 has a 24-month warranty [1]."),
        AIMessage(content="Nothing is free forever [9]."),  # cites a source that does not exist
    ]

    async def scripted(messages, *args, **kwargs):
        return agent_replies.pop(0)

    agent = LangGraphAgent()
    agent.tools_by_name = {"knowledge_search": knowledge_search}
    monkeypatch.setattr(agent.llm_service, "call", scripted)
    graph = await agent.create_graph(checkpointer=InMemorySaver())
    judge = AnswerJudge(ScriptedModel(replies=[_grade_call(), _grade_call(correctness=2, declined=False)]))

    results = {}
    try:
        await evaluate_answers(graph, items, judge, results)
    finally:
        set_rag_pipeline(None)
    report = build_report("t", {}, items, results, {"answers.faithfulness": 4.0, "answers.refusal_accuracy": 1.0})

    assert results["w"].runs[0].answer == "The NovaBook 14 has a 24-month warranty [1]."
    assert results["w"].runs[0].tools_called == ["knowledge_search"]
    assert results["w"].runs[0].trajectory_match is True
    assert results["w"].runs[0].steps == 2
    assert results["adv"].runs[0].refusal_correct is False
    assert report.answers["all"]["trajectory"] == 1.0
    assert report.answers["all"]["mean_tool_calls"] == 0.5
    assert report.answers["all"]["pass_rate"] == 0.5  # the adversarial item should have declined
    assert report.answers["all"]["correctness"] == 3.5
    assert report.answers["all"]["bad_citation_rate"] == 0.0  # [9] in an answer without retrieval is not checked
    assert [g.passed for g in report.gates] == [True, False]
    assert "**FAIL**" in to_markdown(report)


def test_failed_items_count_against_the_answer_gates():
    # An agent that errors on half the questions must not pass on the half that worked.
    items = [GoldenItem(id="ok", question="q1"), GoldenItem(id="boom", question="q2")]
    grade = AnswerGrade(faithfulness=5, correctness=5, citation_support=5, declined=False, rationale="ok")
    results = {
        "ok": ItemResult(
            id="ok", difficulty="unspecified", question="q1", runs=[AnswerRun(answer="fine", grade=grade, passed=True)]
        ),
        "boom": ItemResult(
            id="boom",
            difficulty="unspecified",
            question="q2",
            runs=[AnswerRun(error="RuntimeError: provider down", passed=False)],
        ),
    }

    report = build_report("t", {}, items, results, {"answers.faithfulness": 4.0, "answers.error_rate": 0.0})

    assert report.answers["all"]["error_rate"] == 0.5
    assert report.answers["all"]["n"] == 2
    assert [(g.metric, g.passed) for g in report.gates] == [
        ("answers.faithfulness", True),
        ("answers.error_rate", False),
    ]


async def test_pairwise_judge_reports_position_bias():
    always_a = AIMessage(
        content="", tool_calls=[{"name": "PairVerdict", "args": {"winner": "A", "reason": "first"}, "id": "p"}]
    )
    judge = AnswerJudge(ScriptedModel(replies=[always_a, always_a]))
    item = GoldenItem(id="1", question="q", expected_answer="r")

    assert await judge.compare(item, "answer one", "answer two") == "inconsistent"


async def test_repeated_runs_report_how_often_an_item_passes(monkeypatch):
    # The agent searches on two of three runs and answers from memory on the other.
    items = [GoldenItem(id="w", question="Warranty?", expected_tools=["knowledge_search"])]
    search = AIMessage(content="", tool_calls=[{"name": "knowledge_search", "args": {"query": "warranty"}, "id": "k"}])
    replies = [search, AIMessage(content="24 months."), AIMessage(content="I think 12 months."), search]
    replies.append(AIMessage(content="24 months."))

    async def scripted(messages, *args, **kwargs):
        return replies.pop(0)

    set_rag_pipeline(RAGPipeline(HashingEmbedder(64), InMemoryVectorStore()))
    agent = LangGraphAgent()
    agent.tools_by_name = {"knowledge_search": knowledge_search}
    monkeypatch.setattr(agent.llm_service, "call", scripted)
    graph = await agent.create_graph(checkpointer=InMemorySaver())
    results: dict[str, ItemResult] = {}
    try:
        await evaluate_answers(graph, items, None, results, runs=3)
    finally:
        set_rag_pipeline(None)
    report = build_report("t", {}, items, results, {"answers.pass_rate": 0.9, "answers.pass_all_runs": 1.0})

    assert [run.trajectory_match for run in results["w"].runs] == [True, False, True]
    assert report.answers["all"]["n"] == 3
    assert report.answers["all"]["pass_rate"] == pytest.approx(0.667)
    assert report.answers["all"]["pass_all_runs"] == 0.0
    assert not report.passed
    assert "1/3 runs failed: tools called []" in to_markdown(report)


def test_efficiency_gates_are_upper_limits():
    items = [GoldenItem(id="a", question="q")]
    run = AnswerRun(answer="x", tools_called=["s", "s", "s"], steps=4, tokens=9000, passed=True)
    results = {"a": ItemResult(id="a", difficulty="unspecified", question="q", runs=[run])}

    report = build_report(
        "t", {}, items, results, {"answers.mean_tool_calls": 2, "answers.mean_tokens": 10000, "answers.mean_steps": 4}
    )

    assert [(g.metric, g.passed) for g in report.gates] == [
        ("answers.mean_tool_calls", False),
        ("answers.mean_tokens", True),
        ("answers.mean_steps", True),
    ]
