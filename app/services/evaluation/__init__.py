"""Evaluation: golden sets, retrieval metrics, LLM-as-judge with bias controls, gated reports."""

from app.services.evaluation.dataset import (
    GoldenItem,
    load_golden,
)
from app.services.evaluation.judge import (
    AnswerGrade,
    AnswerJudge,
    cohen_kappa,
)
from app.services.evaluation.retrieval import (
    RetrievalScores,
    score_retrieval,
)
from app.services.evaluation.runner import (
    EvalReport,
    build_report,
    evaluate_answers,
    evaluate_retrieval,
    to_markdown,
)

__all__ = [
    "AnswerGrade",
    "AnswerJudge",
    "EvalReport",
    "GoldenItem",
    "RetrievalScores",
    "build_report",
    "cohen_kappa",
    "evaluate_answers",
    "evaluate_retrieval",
    "load_golden",
    "score_retrieval",
    "to_markdown",
]
