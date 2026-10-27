"""Knowledge-base search tool backed by the RAG pipeline."""

from langchain_core.tools import tool

from app.core.kit import get_rag_pipeline
from app.services.rag import format_context


@tool(response_format="content_and_artifact")
async def knowledge_search(query: str) -> tuple[str, list[str]]:
    """Search the internal knowledge base and return numbered sources to cite.

    Use this for questions about the documents that were ingested into the
    knowledge base. Cite sources in the answer with their labels, e.g. [1].

    Args:
        query: A focused search query in the user's language.
    """
    result = await get_rag_pipeline().retrieve(query)
    # The artifact (ids of the documents used) is kept on the ToolMessage but not
    # shown to the model; the answer cache uses it to invalidate stale answers.
    document_ids = sorted({r.chunk.document_id for r in result.results})
    return format_context(result), document_ids
