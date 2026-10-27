"""Run a golden set against the retrieval pipeline and the real agent graph, then gate on thresholds."""

import time
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Any,
    Optional,
    Sequence,
)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from pydantic import (
    BaseModel,
    Field,
)
import structlog

from app.services.cost_guard import TurnUsage
from app.services.evaluation.dataset import GoldenItem
from app.services.evaluation.judge import (
    AnswerGrade,
    AnswerJudge,
    combine_grades,
    grade_spread,
)
from app.services.evaluation.retrieval import (
    RetrievalScores,
    aggregate,
    score_retrieval,
)
from app.services.evaluation.simulation import (
    ConversationResult,
    summarize_conversations,
)
from app.services.evaluation.trajectory import match_trajectory
from app.services.rag import RAGPipeline
from app.services.rag.citations import (
    check_answer,
    cited_labels,
)

logger = structlog.get_logger()


def _text(content: str | list) -> str:
    """Plain text of a message content (a string, or content blocks from reasoning models)."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


class AnswerRun(BaseModel):
    """One run of the agent on a golden item, and how it was graded."""

    answer: Optional[str] = None
    grade: Optional[AnswerGrade] = None
    judge_spread: Optional[int] = Field(default=None, description="Score gap between judge samples (samples > 1)")
    citations_out_of_range: list[int] = Field(default_factory=list)
    refusal_correct: Optional[bool] = None
    tools_called: list[str] = Field(default_factory=list, description="Tool names in call order")
    trajectory_match: Optional[bool] = None
    steps: int = Field(default=0, description="Model replies the agent produced in the run")
    tokens: int = Field(default=0, description="Input plus output tokens of every model call in the run")
    latency_s: Optional[float] = None
    error: Optional[str] = None
    passed: Optional[bool] = None


class ItemResult(BaseModel):
    """Everything measured for one golden item: retrieval once, then each agent run."""

    id: str
    difficulty: str
    question: str
    retrieved: list[str] = Field(default_factory=list)
    retrieval: Optional[RetrievalScores] = None
    runs: list[AnswerRun] = Field(default_factory=list)


class Gate(BaseModel):
    """One threshold check."""

    metric: str
    value: float
    threshold: float
    passed: bool


class EvalReport(BaseModel):
    """A full evaluation run."""

    created_at: str
    dataset: str
    config: dict[str, Any]
    retrieval: dict[str, dict[str, float]] = Field(default_factory=dict)
    answers: dict[str, dict[str, float]] = Field(default_factory=dict)
    conversations: dict[str, dict[str, float]] = Field(default_factory=dict)
    gates: list[Gate] = Field(default_factory=list)
    items: list[ItemResult] = Field(default_factory=list)
    conversation_results: list[ConversationResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when every gate passed."""
        return all(g.passed for g in self.gates)


async def evaluate_retrieval(pipeline: RAGPipeline, items: list[GoldenItem], k: int) -> dict[str, ItemResult]:
    """Retrieve for each question and score against its expected documents."""
    results: dict[str, ItemResult] = {}
    for item in items:
        retrieval = await pipeline.retrieve(item.question, top_k=k)
        docs = [r.chunk.document_id for r in retrieval.results]
        results[item.id] = ItemResult(
            id=item.id,
            difficulty=item.difficulty,
            question=item.question,
            retrieved=docs,
            retrieval=score_retrieval(docs, item.expected_sources, k),
        )
    return results


def run_passed(run: AnswerRun, item: GoldenItem, pass_score: int = 4) -> bool:
    """Whether one run did everything right.

    That is: no error, the expected tools, valid citations, a correct refusal when one
    was due, and judge scores of at least ``pass_score``.
    """
    if run.error is not None or run.trajectory_match is False or run.citations_out_of_range:
        return False
    if item.should_refuse and run.refusal_correct is False:
        return False
    if run.grade is not None and min(run.grade.faithfulness, run.grade.correctness) < pass_score:
        return False
    return True


async def _run_once(graph: Any, item: GoldenItem, judge: Optional[AnswerJudge], pass_score: int) -> AnswerRun:
    run = AnswerRun()
    usage = TurnUsage()
    started = time.perf_counter()
    try:
        state = await graph.ainvoke(
            {"messages": [HumanMessage(content=item.question)]},
            {"configurable": {"thread_id": f"eval-{item.id}-{time.time_ns()}"}, "callbacks": [usage]},
        )
    except Exception as e:
        run.error = f"{type(e).__name__}: {e}"
        run.passed = False
        logger.warning("eval_item_failed", item=item.id, error=run.error)
        return run
    run.latency_s = round(time.perf_counter() - started, 3)
    messages = state["messages"]
    replies = [m for m in messages if isinstance(m, AIMessage)]
    run.steps = len(replies)
    run.tokens = sum(input_tokens + output_tokens for _, input_tokens, output_tokens in usage.calls)
    run.tools_called = [call["name"] for m in replies for call in m.tool_calls]
    if item.expected_tools:
        run.trajectory_match = match_trajectory(run.tools_called, item.expected_tools, item.tool_match)
    final = messages[-1]
    run.answer = _text(final.content) if isinstance(final, AIMessage) else ""
    contexts = [_text(m.content) for m in messages if isinstance(m, ToolMessage) and m.name == "knowledge_search"]
    source_count = max((max(cited_labels(c), default=0) for c in contexts), default=0)
    run.citations_out_of_range = check_answer(run.answer, source_count).out_of_range if source_count else []
    if judge is not None:
        try:
            grades = await judge.grade_samples(item, run.answer, "\n\n".join(contexts))
            run.grade = combine_grades(grades)
            if len(grades) > 1:
                run.judge_spread = grade_spread(grades)
            if item.should_refuse:
                run.refusal_correct = run.grade.declined
        except Exception as e:
            run.error = f"judge failed: {type(e).__name__}: {e}"
    run.passed = run_passed(run, item, pass_score)
    return run


async def evaluate_answers(
    graph: Any,
    items: list[GoldenItem],
    judge: Optional[AnswerJudge],
    results: dict[str, ItemResult],
    runs: int = 1,
    pass_score: int = 4,
) -> None:
    """Ask the real agent graph each question ``runs`` times (fresh thread per run) and grade every answer.

    An agent does not answer the same way twice; with ``runs > 1`` the report shows how
    often it gets an item right (``pass_rate``) and how many items it gets right every
    time (``pass_all_runs``), not one lucky or unlucky sample.
    """
    if runs < 1:
        raise ValueError("runs must be at least 1")
    for item in items:
        result = results.setdefault(
            item.id, ItemResult(id=item.id, difficulty=item.difficulty, question=item.question)
        )
        for _ in range(runs):
            result.runs.append(await _run_once(graph, item, judge, pass_score))


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 3)


def summarize_answers(items: list[GoldenItem], results: dict[str, ItemResult]) -> dict[str, dict[str, float]]:
    """Mean judge scores overall and per difficulty, plus error, refusal, citation, tool and pass rates.

    Every run counts once. Runs that failed (agent or judge error) count in ``n`` and
    ``error_rate``: averaging only the runs that worked would let a half-broken agent
    pass every score gate.
    """
    groups: dict[str, list[ItemResult]] = {"all": []}
    for item in items:
        r = results.get(item.id)
        if r is None or not r.runs:
            continue  # retrieval-only run: no answer was attempted
        groups["all"].append(r)
        groups.setdefault(item.difficulty, []).append(r)

    summary: dict[str, dict[str, float]] = {}
    if not groups["all"]:
        return summary
    for name, item_results in groups.items():
        rs = [run for r in item_results for run in r.runs]
        answered = [r for r in rs if r.answer is not None]
        graded = [r for r in rs if r.grade]
        refusals = [r for r in rs if r.refusal_correct is not None]
        trajectories = [r for r in rs if r.trajectory_match is not None]
        row: dict[str, float] = {"n": len(rs), "error_rate": _mean([r.error is not None for r in rs])}
        if graded:
            for metric in ("faithfulness", "correctness", "citation_support"):
                row[metric] = _mean([getattr(r.grade, metric) for r in graded])
        spreads = [r.judge_spread for r in rs if r.judge_spread is not None]
        if spreads:
            row["judge_spread"] = _mean(spreads)
        if refusals:
            row["refusal_accuracy"] = _mean([bool(r.refusal_correct) for r in refusals])
        if trajectories:
            row["trajectory"] = _mean([bool(r.trajectory_match) for r in trajectories])
        if answered:
            row["bad_citation_rate"] = _mean([bool(r.citations_out_of_range) for r in answered])
            row["mean_tool_calls"] = _mean([len(r.tools_called) for r in answered])
            row["mean_steps"] = _mean([r.steps for r in answered])
            if any(r.tokens for r in answered):
                row["mean_tokens"] = _mean([r.tokens for r in answered])
            latencies = sorted(r.latency_s for r in answered if r.latency_s is not None)
            if latencies:
                row["p95_latency_s"] = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
        row["pass_rate"] = _mean([bool(r.passed) for r in rs])
        if any(len(r.runs) > 1 for r in item_results):
            row["pass_all_runs"] = _mean([all(run.passed for run in r.runs) for r in item_results])
        summary[name] = row
    return summary


LOWER_IS_BETTER = {
    "error_rate",
    "bad_citation_rate",
    "p95_latency_s",
    "mean_tool_calls",
    "mean_steps",
    "mean_tokens",
    "turn_limit_rate",
    "mean_turns",
}


def apply_gates(report: EvalReport, thresholds: dict[str, float]) -> None:
    """Add a gate for each threshold whose metric was measured (``retrieval.hit``, ``answers.faithfulness``, ...)."""
    for key, threshold in thresholds.items():
        section, _, metric = key.partition(".")
        value = getattr(report, section, {}).get("all", {}).get(metric)
        if value is None:
            continue
        passed = value <= threshold if metric in LOWER_IS_BETTER else value >= threshold
        report.gates.append(Gate(metric=key, value=value, threshold=threshold, passed=passed))


def build_report(
    dataset: str,
    config: dict[str, Any],
    items: list[GoldenItem],
    results: dict[str, ItemResult],
    thresholds: dict[str, float],
    conversations: Optional[list[ConversationResult]] = None,
) -> EvalReport:
    """Aggregate per-item (and simulated conversation) results into a report and evaluate the gates."""
    retrieval_scores = {i: r.retrieval for i, r in results.items() if r.retrieval is not None}
    report = EvalReport(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset=dataset,
        config=config,
        retrieval=aggregate(retrieval_scores, {i.id: i.difficulty for i in items}) if retrieval_scores else {},
        answers=summarize_answers(items, results),
        items=[results[i.id] for i in items if i.id in results],
        conversations=summarize_conversations(conversations or []),
        conversation_results=conversations or [],
    )
    apply_gates(report, thresholds)
    return report


def to_markdown(report: EvalReport) -> str:
    """Human-readable summary for a PR comment or a report file."""
    lines = [f"# Evaluation report — {report.dataset}", "", f"Run at {report.created_at}.", ""]
    if report.gates:
        lines += ["| Gate | Value | Threshold | Result |", "|---|---|---|---|"]
        lines += [
            f"| `{g.metric}` | {g.value} | {g.threshold} | {'pass' if g.passed else '**FAIL**'} |"
            for g in report.gates
        ]
        lines.append("")
    for title, table in (
        ("Retrieval", report.retrieval),
        ("Answers", report.answers),
        ("Conversations", report.conversations),
    ):
        if not table:
            continue
        metrics = sorted({m for row in table.values() for m in row})
        lines += [f"## {title}", "", "| Group | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
        for group, row in table.items():
            lines.append(f"| {group} | " + " | ".join(str(row.get(m, "")) for m in metrics) + " |")
        lines.append("")
    lines += _items_to_look_at(report.items)
    lines += _conversations_to_look_at(report.conversation_results)
    return "\n".join(lines) + "\n"


def _why_run_failed(run: AnswerRun) -> str:
    if run.error:
        return run.error
    if run.trajectory_match is False:
        return f"tools called {run.tools_called}"
    if run.citations_out_of_range:
        return f"cites sources that do not exist {run.citations_out_of_range}"
    if run.refusal_correct is False:
        return "should have declined"
    return run.grade.rationale if run.grade else "failed"


def _items_to_look_at(items: list[ItemResult]) -> list[str]:
    lines: list[str] = []
    for r in items:
        failed = [run for run in r.runs if run.passed is False]
        missed = r.retrieval is not None and r.retrieval.hit == 0
        if not failed and not missed:
            continue
        if not lines:
            lines = ["## Items to look at", ""]
        if failed:
            why = f"{len(failed)}/{len(r.runs)} runs failed: {_why_run_failed(failed[0])}"
        else:
            why = f"expected sources not retrieved (got {r.retrieved[:3]})"
        lines.append(f"- **{r.id}** ({r.difficulty}) {r.question} — {why}")
    return lines


def _conversations_to_look_at(results: list[ConversationResult]) -> list[str]:
    failed = [r for r in results if r.success is not True]
    if not failed:
        return []
    lines = ["", "## Conversations to look at", ""]
    for r in failed:
        why = r.error or r.rationale or r.stop_reason
        lines.append(f"- **{r.scenario_id}** ({r.turns} turns, {r.stop_reason}) — {why}")
    return lines
