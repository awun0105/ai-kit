"""Ingest a folder of Markdown/text files into the RAG knowledge base.

Usage:
    APP_ENV=development uv run python -m scripts.ingest path/to/corpus
    APP_ENV=development uv run python -m scripts.ingest path/to/corpus --query "test question"
"""

import argparse
import asyncio
from pathlib import Path

from app.services.rag import (
    build_pipeline_from_settings,
    format_context,
)
from app.services.reliability import build_answer_cache_from_settings
from app.services.rag.chunking import load_documents


async def main() -> None:
    """Load documents, ingest them, and optionally run one query."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", type=Path, help="Folder containing .md/.txt files")
    parser.add_argument("--query", help="Run one retrieval after ingesting and print the context")
    args = parser.parse_args()

    pipeline = build_pipeline_from_settings()
    try:
        report = await pipeline.ingest(load_documents(args.corpus))
        print(report.model_dump_json(indent=2))
        cache = build_answer_cache_from_settings()
        if cache is not None and report.changed_documents:
            # Cached answers built on documents that just changed must not be served again.
            removed = await cache.invalidate(report.changed_documents)
            print(f"answer cache: removed {removed} stale answers")
            close_cache = getattr(cache.store, "close", None)
            if close_cache is not None:
                await close_cache()
        if args.query:
            print(format_context(await pipeline.retrieve(args.query)))
    finally:
        close = getattr(pipeline.store, "close", None)
        if close is not None:
            await close()


if __name__ == "__main__":
    asyncio.run(main())
