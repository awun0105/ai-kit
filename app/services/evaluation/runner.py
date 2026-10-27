"""Run a golden set against the retrieval pipeline and the real agent graph, then gate on thresholds."""

import time
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Any,
    Optional,
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

from app.core.logging import logger
from app.services.evaluation.dataset import GoldenItem
from app.services.evaluation.judge import (
    AnswerGrade,
    AnswerJudge,
)
from app.services.evaluation.retrieval import (
    RetrievalScores,
    aggregate,
    score_retrieval,
)
from app.services.rag import RAGPipeline
from app.services.rag.citations import (
    check_answer,
    cited_labels,
)
from app.utils import extract_text_content


class ItemResult(BaseModel):
    """Everything measured for one golden item."""

    id: str
    difficulty: str
    question: str
    retrieved: list[str] = Field(default_factory=list)
    retrieval: Optional[RetrievalScores] = None
    answer: Optional[str] = None
    grade: Optional[AnswerGrade] = None
    citations_out_of_range: list[int] = Field(default_factory=list)
    refusal_correct: Optional[bool] = None
    latency_s: Optional[float] = None
    error: Optional[str] = None


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
    gates: list[Gate] = Field(default_factory=list)
    items: list[ItemResult] = Field(default_factory=list)

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


async def evaluate_answers(
    graph: Any, items: list[GoldenItem], judge: Optional[AnswerJudge], results: dict[str, ItemResult]
) -> None:
    """Ask the real agent graph each question (fresh thread per item) and grade the answer."""
    for item in items:
        result = results.setdefault(
            item.id, ItemResult(id=item.id, difficulty=item.difficulty, question=item.question)
        )
        started = time.perf_counter()
        try:
            state = await graph.ainvoke(
                {"messages": [HumanMessage(content=item.question)]},
                {"configurable": {"thread_id": f"eval-{item.id}-{time.time_ns()}"}},
            )
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            logger.warning("eval_item_failed", item=item.id, error=result.error)
            continue
        result.latency_s = round(time.perf_counter() - started, 3)
        messages = state["messages"]
        final = messages[-1]
        result.answer = extract_text_content(final.content) if isinstance(final, AIMessage) else ""
        contexts = [
            extract_text_content(m.content)
            for m in messages
            if isinstance(m, ToolMessage) and m.name == "knowledge_search"
        ]
        source_count = max((max(cited_labels(c), default=0) for c in contexts), default=0)
        result.citations_out_of_range = check_answer(result.answer, source_count).out_of_range if source_count else []
        if judge is not None:
            try:
                result.grade = await judge.grade(item, result.answer, "\n\n".join(contexts))
                if item.should_refuse:
                    result.refusal_correct = result.grade.declined
            except Exception as e:
                result.error = f"judge failed: {type(e).__name__}: {e}"


def summarize_answers(items: list[GoldenItem], results: dict[str, ItemResult]) -> dict[str, dict[str, float]]:
    """Mean judge scores overall and per difficulty, plus error, refusal and citation rates.

    Items whose agent run or grading failed count in ``n`` and ``error_rate``: averaging
    only the items that worked would let a half-broken agent pass every score gate.
    """
    groups: dict[str, list[ItemResult]] = {"all": []}
    for item in items:
        r = results.get(item.id)
        if r is None or (r.answer is None and r.error is None):
            continue  # retrieval-only run: no answer was attempted
        groups["all"].append(r)
        groups.setdefault(item.difficulty, []).append(r)

    summary: dict[str, dict[str, float]] = {}
    if not groups["all"]:
        return summary
    for name, rs in groups.items():
        answered = [r for r in rs if r.answer is not None]
        graded = [r for r in rs if r.grade]
        refusals = [r for r in rs if r.refusal_correct is not None]
        row: dict[str, float] = {"n": len(rs), "error_rate": round(sum(r.error is not None for r in rs) / len(rs), 3)}
        if graded:
            for metric in ("faithfulness", "correctness", "citation_support"):
                row[metric] = round(sum(getattr(r.grade, metric) for r in graded) / len(graded), 3)  # type: ignore[union-attr]
        if refusals:
            row["refusal_accuracy"] = round(sum(bool(r.refusal_correct) for r in refusals) / len(refusals), 3)
        if answered:
            row["bad_citation_rate"] = round(sum(bool(r.citations_out_of_range) for r in answered) / len(answered), 3)
            latencies = sorted(r.latency_s for r in answered if r.latency_s is not None)
            if latencies:
                row["p95_latency_s"] = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
        summary[name] = row
    return summary


def apply_gates(report: EvalReport, thresholds: dict[str, float]) -> None:
    """Add a gate for each threshold whose metric was measured (``retrieval.hit``, ``answers.faithfulness``, ...)."""
    for key, threshold in thresholds.items():
        section, _, metric = key.partition(".")
        value = getattr(report, section, {}).get("all", {}).get(metric)
        if value is None:
            continue
        lower_is_better = metric in ("error_rate", "bad_citation_rate", "p95_latency_s")
        passed = value <= threshold if lower_is_better else value >= threshold
        report.gates.append(Gate(metric=key, value=value, threshold=threshold, passed=passed))


def build_report(
    dataset: str,
    config: dict[str, Any],
    items: list[GoldenItem],
    results: dict[str, ItemResult],
    thresholds: dict[str, float],
) -> EvalReport:
    """Aggregate per-item results into a report and evaluate the gates."""
    retrieval_scores = {i: r.retrieval for i, r in results.items() if r.retrieval is not None}
    report = EvalReport(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset=dataset,
        config=config,
        retrieval=aggregate(retrieval_scores, {i.id: i.difficulty for i in items}) if retrieval_scores else {},
        answers=summarize_answers(items, results),
        items=[results[i.id] for i in items if i.id in results],
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
    for title, table in (("Retrieval", report.retrieval), ("Answers", report.answers)):
        if not table:
            continue
        metrics = sorted({m for row in table.values() for m in row})
        lines += [f"## {title}", "", "| Group | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
        for group, row in table.items():
            lines.append(f"| {group} | " + " | ".join(str(row.get(m, "")) for m in metrics) + " |")
        lines.append("")
    failures = [
        r
        for r in report.items
        if r.error
        or (r.retrieval and r.retrieval.hit == 0)
        or (r.grade and min(r.grade.faithfulness, r.grade.correctness) <= 2)
        or r.refusal_correct is False
    ]
    if failures:
        lines += ["## Items to look at", ""]
        for r in failures:
            why = r.error or (
                r.grade.rationale if r.grade else f"expected sources not retrieved (got {r.retrieved[:3]})"
            )
            lines.append(f"- **{r.id}** ({r.difficulty}) {r.question} — {why}")
    return "\n".join(lines) + "\n"
