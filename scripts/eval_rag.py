"""Evaluate retrieval, and optionally the agent's answers, on a golden set. Exits 1 when a gate fails.

Offline retrieval regression check (no API key; runs in CI):
    APP_ENV=test uv run python -m scripts.eval_rag --retrieval-only --embedding hashing

Full run: the real agent graph answers every question, a separate judge model
(EVALUATION_LLM / EVALUATION_BASE_URL / EVALUATION_API_KEY) grades the answers:
    APP_ENV=development uv run python -m scripts.eval_rag

Evaluate the knowledge base the app already serves instead of a fresh copy of the sample corpus:
    APP_ENV=development uv run python -m scripts.eval_rag --dataset my_golden.jsonl --use-knowledge-base
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
    build_report,
    evaluate_answers,
    evaluate_retrieval,
    load_golden,
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
    p.add_argument("--max-error-rate", type=float, default=0.0, help="Share of items whose run or grading failed")
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
    items = load_golden(args.dataset)
    pipeline = await _pipeline(args)
    results = await evaluate_retrieval(pipeline, items, args.k)
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
        await evaluate_answers(graph, items, AnswerJudge(judge_model), results)
        thresholds |= {
            "answers.faithfulness": args.min_faithfulness,
            "answers.correctness": args.min_correctness,
            "answers.refusal_accuracy": args.min_refusal,
            "answers.bad_citation_rate": args.max_bad_citations,
            "answers.error_rate": args.max_error_rate,
        }

    config = {
        "k": args.k,
        "embedding": "knowledge-base" if args.use_knowledge_base else args.embedding,
        "retrieval_only": args.retrieval_only,
        "judge": None if args.retrieval_only else settings.EVALUATION_LLM,
    }
    report = build_report(str(args.dataset), config, items, results, thresholds)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (args.out / f"eval-{stamp}.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown = to_markdown(report)
    (args.out / f"eval-{stamp}.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
