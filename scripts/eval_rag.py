"""Evaluate retrieval, and optionally the agent's answers, on a golden set. Exits 1 when a gate fails.

Offline retrieval regression check (no API key; runs in CI):
    APP_ENV=test uv run python -m scripts.eval_rag --retrieval-only --embedding hashing

Full run: the real agent graph answers every question, a separate judge model
(EVALUATION_LLM / EVALUATION_BASE_URL / EVALUATION_API_KEY) grades the answers:
    APP_ENV=development uv run python -m scripts.eval_rag

Evaluate the knowledge base the app already serves instead of a fresh copy of the sample corpus:
    APP_ENV=development uv run python -m scripts.eval_rag --dataset my_golden.jsonl --use-knowledge-base

Add multi-turn conversations with a simulated user (same model as the judge):
    APP_ENV=development uv run python -m scripts.eval_rag --scenarios evals/datasets/orbittech/scenarios.json

Run every question 3 times and gate on how often, and how consistently, the agent gets it right:
    APP_ENV=development uv run python -m scripts.eval_rag --runs 3 --min-pass-rate 0.8 --min-pass-all-runs 0.7
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import settings
from app.core.kit import (
    build_pipeline_from_settings,
    set_rag_pipeline,
)
from app.core.langgraph.graph import LangGraphAgent
from app.core.langgraph.tools.knowledge_search import knowledge_search
from app.services.evaluation import (
    AnswerJudge,
    ConversationResult,
    TaskJudge,
    UserSimulator,
    build_report,
    evaluate_answers,
    evaluate_retrieval,
    load_golden,
    load_scenarios,
    simulate_conversation,
    to_markdown,
)
from app.services.rag import (
    HashingEmbedder,
    InMemoryVectorStore,
    OpenAIEmbedder,
    RAGPipeline,
)
from app.services.rag.chunking import load_documents

SAMPLE = Path(__file__).resolve().parent.parent / "evals" / "datasets" / "orbittech"


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=SAMPLE / "golden.json")
    p.add_argument("--corpus", type=Path, default=SAMPLE / "corpus", help="Ingested into a throwaway in-memory store")
    p.add_argument("--use-knowledge-base", action="store_true", help="Query the configured RAG store instead")
    p.add_argument("--embedding", choices=["configured", "hashing"], default="configured")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--min-hit", type=float, default=0.9)
    p.add_argument("--min-recall", type=float, default=0.85)
    p.add_argument("--min-mrr", type=float, default=0.75)
    p.add_argument("--min-faithfulness", type=float, default=4.0)
    p.add_argument("--min-correctness", type=float, default=3.5)
    p.add_argument("--min-refusal", type=float, default=0.66)
    p.add_argument("--max-bad-citations", type=float, default=0.1)
    p.add_argument("--max-error-rate", type=float, default=0.0, help="Share of runs whose agent or judge failed")
    p.add_argument("--scenarios", type=Path, help="Simulated multi-turn conversations (JSON list or JSONL)")
    p.add_argument("--min-conversation-success", type=float, default=0.8, help="With --scenarios")
    p.add_argument("--runs", type=int, default=1, help="Times the agent answers each question")
    p.add_argument("--judge-samples", type=int, default=1, help="Grades per answer, combined by median and majority")
    p.add_argument("--pass-score", type=int, default=4, help="Judge score a run needs to pass")
    p.add_argument("--min-trajectory", type=float, default=0.9, help="Share of runs that called the expected tools")
    p.add_argument("--min-pass-rate", type=float, default=0.0, help="Share of runs that passed (0 = no gate)")
    p.add_argument(
        "--min-pass-all-runs", type=float, default=0.0, help="Share of items passed every run (0 = no gate)"
    )
    p.add_argument("--max-mean-tokens", type=float, default=0.0, help="Tokens per answer (0 = no gate)")
    p.add_argument("--max-mean-tool-calls", type=float, default=0.0, help="Tool calls per answer (0 = no gate)")
    return p.parse_args()


async def _pipeline(args: argparse.Namespace) -> RAGPipeline:
    if args.use_knowledge_base:
        return build_pipeline_from_settings()
    embedder = (
        HashingEmbedder(1024)
        if args.embedding == "hashing"
        else OpenAIEmbedder(
            settings.RAG_EMBEDDING_MODEL,
            settings.RAG_EMBEDDING_DIM,
            settings.RAG_EMBEDDING_API_KEY,
            settings.RAG_EMBEDDING_BASE_URL or None,
        )
    )
    pipeline = RAGPipeline(
        embedder=embedder,
        store=InMemoryVectorStore(),
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        score_threshold=settings.RAG_SCORE_THRESHOLD,
    )
    await pipeline.ingest(load_documents(args.corpus))
    return pipeline


async def main() -> int:
    """Run the evaluation, write JSON and Markdown reports, and return the exit code."""
    args = _args()
    items = load_golden(args.dataset, retrieval_tool=knowledge_search.name)
    pipeline = await _pipeline(args)
    results = await evaluate_retrieval(pipeline, items, args.k)
    conversations: list[ConversationResult] = []
    thresholds = {
        "retrieval.hit": args.min_hit,
        "retrieval.recall": args.min_recall,
        "retrieval.reciprocal_rank": args.min_mrr,
    }

    if not args.retrieval_only:
        set_rag_pipeline(pipeline)
        agent = LangGraphAgent()
        if "knowledge_search" not in agent.tools_by_name:
            agent.register_tools([knowledge_search])
        graph = await agent.create_graph(checkpointer=InMemorySaver())
        judge_model = init_chat_model(
            settings.EVALUATION_LLM,
            model_provider="openai",
            base_url=settings.EVALUATION_BASE_URL,
            api_key=settings.EVALUATION_API_KEY,
        )
        await evaluate_answers(
            graph,
            items,
            AnswerJudge(judge_model, samples=args.judge_samples),
            results,
            runs=args.runs,
            pass_score=args.pass_score,
        )
        thresholds |= {
            "answers.faithfulness": args.min_faithfulness,
            "answers.correctness": args.min_correctness,
            "answers.refusal_accuracy": args.min_refusal,
            "answers.bad_citation_rate": args.max_bad_citations,
            "answers.error_rate": args.max_error_rate,
            "answers.trajectory": args.min_trajectory,
        }
        optional = {
            "answers.pass_rate": args.min_pass_rate,
            "answers.pass_all_runs": args.min_pass_all_runs,
            "answers.mean_tokens": args.max_mean_tokens,
            "answers.mean_tool_calls": args.max_mean_tool_calls,
        }
        thresholds |= {metric: value for metric, value in optional.items() if value > 0}
        if args.scenarios:
            simulator, task_judge = UserSimulator(judge_model), TaskJudge(judge_model)
            for scenario in load_scenarios(args.scenarios):
                for _ in range(args.runs):
                    conversations.append(await simulate_conversation(graph, simulator, scenario, task_judge))
            thresholds["conversations.success_rate"] = args.min_conversation_success

    config = {
        "k": args.k,
        "embedding": "knowledge-base" if args.use_knowledge_base else args.embedding,
        "retrieval_only": args.retrieval_only,
        "runs": 0 if args.retrieval_only else args.runs,
        "judge": None if args.retrieval_only else settings.EVALUATION_LLM,
        "judge_samples": args.judge_samples,
        "scenarios": str(args.scenarios) if args.scenarios and not args.retrieval_only else None,
    }
    report = build_report(str(args.dataset), config, items, results, thresholds, conversations)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (args.out / f"eval-{stamp}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown = to_markdown(report)
    (args.out / f"eval-{stamp}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
