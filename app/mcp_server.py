"""Expose the kit's knowledge base as an MCP server.

Lets other agents (Claude Code, Cursor, another LangGraph app) search the same
documents the chat agent uses, with the same hybrid retrieval and citations.

    # local, over stdio (e.g. in a Claude Code / Cursor MCP config)
    APP_ENV=development uv run python -m app.mcp_server

    # over HTTP; refuses to start without MCP_SERVER_TOKEN
    APP_ENV=development uv run python -m app.mcp_server --http --port 8765
"""

import argparse
from typing import Any, Optional

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

from app.core.config import settings
from app.core.kit import (
    get_guardrails,
    get_rag_pipeline,
)
from app.services.rag import format_context


def build_server(token: Optional[str] = None) -> FastMCP:
    """Create the server; with ``token`` every request must send ``Authorization: Bearer <token>``."""
    auth = StaticTokenVerifier(tokens={token: {"client_id": "ai-kit-mcp", "scopes": []}}) if token else None
    mcp = FastMCP(
        "ai-kit",
        instructions="Search the organisation's knowledge base. Cite results with their [n] labels.",
        auth=auth,
    )

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": False})
    async def knowledge_search(query: str, top_k: int = 5) -> dict[str, Any]:
        """Hybrid (dense + BM25) search over the ingested documents.

        Returns numbered sources for citation plus the matching chunks.
        """
        result = await get_rag_pipeline().retrieve(query, top_k=max(1, min(top_k, 20)))
        guard = get_guardrails()
        return {
            # The same output rail as the chat agent: no PII or secrets leave through MCP either.
            "context": guard.check_output(format_context(result)).text,
            "low_confidence": result.low_confidence,
            "sources": [
                {
                    "label": n,
                    "document_id": r.chunk.document_id,
                    "title": r.chunk.title,
                    "url": r.chunk.url,
                    "score": r.score,
                }
                for n, r in enumerate(result.results, start=1)
            ],
        }

    return mcp


def main() -> None:
    """Run over stdio by default, or over streamable HTTP with ``--http``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--http", action="store_true", help="Serve streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.http:
        if not settings.MCP_SERVER_TOKEN:
            raise SystemExit("Refusing to serve the knowledge base over HTTP without MCP_SERVER_TOKEN.")
        build_server(settings.MCP_SERVER_TOKEN).run(transport="http", host=args.host, port=args.port)
    else:
        build_server().run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
