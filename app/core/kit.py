"""Wiring: build the toolbox components from the app's settings and keep one of each per process.

The modules under ``app/services`` (rag, guardrails, reliability, evaluation, mcp,
a2a, agents, cost_guard) take everything they need as constructor arguments and
never read ``settings``, so each can be copied into another project on its own.
This module is the one place that reads the ``RAG_*``, ``GUARD_*``, ``ANSWER_CACHE_*``,
``COST_*`` and ``MCP_*`` settings, assembles the components and hands out the
process-wide instances. Tests replace an instance with the matching ``set_*``.
"""

from typing import (
    Literal,
    Optional,
    cast,
    get_args,
)
from urllib.parse import quote_plus

from langgraph.constants import TAG_NOSTREAM

from app.core.cache import cache_service
from app.core.config import settings
from app.core.prompts import _SYSTEM_PROMPT_TEMPLATE
from app.services.cost_guard import (
    CostGuard,
    parse_prices,
)
from app.services.guardrails import (
    Guardrails,
    LLMPolicyRail,
    PIIAction,
    PolicyVerdict,
)
from app.services.llm import llm_service
from app.services.mcp import (
    MCPToolkit,
    load_mcp_config,
)
from app.services.rag import (
    Embedder,
    HashingEmbedder,
    InMemoryVectorStore,
    OpenAIEmbedder,
    PgVectorStore,
    RAGPipeline,
    Reranker,
    RerankersReranker,
    TokenizerMode,
    VectorStore,
)
from app.services.reliability import (
    AnswerCache,
    InMemoryAnswerStore,
    PgVectorAnswerStore,
    scope_key,
)


def _postgres_url() -> str:
    return (
        f"postgresql://{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )


# ---------------------------------------------------------------- RAG

_pipeline: Optional[RAGPipeline] = None


def build_pipeline_from_settings() -> RAGPipeline:
    """Assemble a pipeline from the RAG_* settings."""
    embedder: Embedder
    if settings.RAG_EMBEDDING_PROVIDER == "hashing":
        embedder = HashingEmbedder(settings.RAG_EMBEDDING_DIM)
    else:
        embedder = OpenAIEmbedder(
            model=settings.RAG_EMBEDDING_MODEL,
            dimension=settings.RAG_EMBEDDING_DIM,
            api_key=settings.RAG_EMBEDDING_API_KEY,
            base_url=settings.RAG_EMBEDDING_BASE_URL or None,
        )

    store: VectorStore
    if settings.RAG_VECTOR_STORE == "memory":
        store = InMemoryVectorStore()
    else:
        store = PgVectorStore.from_settings(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            db=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            table=settings.RAG_TABLE,
            dimension=settings.RAG_EMBEDDING_DIM,
        )

    reranker: Optional[Reranker] = None
    if settings.RAG_RERANKER:
        reranker = RerankersReranker(settings.RAG_RERANKER, settings.RAG_RERANKER_TYPE or None)

    tokenizer = cast(TokenizerMode, settings.RAG_LEXICAL_TOKENIZER)
    if tokenizer not in get_args(TokenizerMode):
        raise ValueError(f"RAG_LEXICAL_TOKENIZER must be one of {get_args(TokenizerMode)}, got {tokenizer!r}")

    return RAGPipeline(
        embedder=embedder,
        store=store,
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
        top_k=settings.RAG_TOP_K,
        score_threshold=settings.RAG_SCORE_THRESHOLD,
        tokenizer=tokenizer,
        reranker=reranker,
        rerank_candidates=settings.RAG_RERANK_CANDIDATES,
    )


def get_rag_pipeline() -> RAGPipeline:
    """Return the process-wide pipeline, building it on first use."""
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline_from_settings()
    return _pipeline


def set_rag_pipeline(pipeline: Optional[RAGPipeline]) -> None:
    """Replace the process-wide pipeline (used by tests)."""
    global _pipeline
    _pipeline = pipeline


# ---------------------------------------------------------------- guardrails

_guardrails: Optional[Guardrails] = None


def build_guardrails_from_settings() -> Guardrails:
    """Assemble the rails from the GUARD_* settings."""
    pii_action = settings.GUARD_INPUT_PII_ACTION
    if pii_action not in ("block", "redact", "allow"):
        raise ValueError(f"GUARD_INPUT_PII_ACTION must be block, redact or allow, got {pii_action!r}")
    fail_mode = settings.GUARD_LLM_RAIL_FAIL_MODE
    if fail_mode not in ("open", "closed"):
        raise ValueError(f"GUARD_LLM_RAIL_FAIL_MODE must be open or closed, got {fail_mode!r}")

    llm_rail = None
    if settings.GUARD_LLM_RAIL_ENABLED:
        model = settings.GUARD_LLM_RAIL_MODEL or None

        async def classify(messages: list[dict], schema: type[PolicyVerdict]) -> PolicyVerdict:
            # nostream: the verdict must never reach a client reading the graph's message stream.
            return await llm_service.call(
                messages, model_name=model, response_format=schema, config={"tags": [TAG_NOSTREAM]}
            )

        llm_rail = LLMPolicyRail(
            settings.GUARD_LLM_RAIL_POLICY, classify, fail_mode=cast(Literal["open", "closed"], fail_mode)
        )

    return Guardrails(
        enabled=settings.GUARD_ENABLED,
        block_injection=settings.GUARD_BLOCK_INJECTION,
        input_pii_action=cast(PIIAction, pii_action),
        output_redaction=settings.GUARD_OUTPUT_REDACTION,
        pii_allowlist=tuple(settings.GUARD_PII_ALLOWLIST),
        denylist=tuple(settings.GUARD_DENYLIST),
        approval_tools=frozenset(settings.GUARD_APPROVAL_TOOLS),
        egress_allowlist=tuple(settings.GUARD_EGRESS_ALLOWLIST),
        screen_tool_output=settings.GUARD_SCREEN_TOOL_OUTPUT,
        llm_rail=llm_rail,
        refusal_message=settings.GUARD_REFUSAL_MESSAGE,
    )


def get_guardrails() -> Guardrails:
    """Return the process-wide rails, building them on first use."""
    global _guardrails
    if _guardrails is None:
        _guardrails = build_guardrails_from_settings()
    return _guardrails


def set_guardrails(guardrails: Optional[Guardrails]) -> None:
    """Replace the process-wide rails (used by tests)."""
    global _guardrails
    _guardrails = guardrails


# ---------------------------------------------------------------- answer cache

_answer_cache: Optional[AnswerCache] = None
_answer_cache_built = False


def build_answer_cache_from_settings() -> Optional[AnswerCache]:
    """Return the configured answer cache, or None when disabled or RAG is off."""
    if not (settings.ANSWER_CACHE_ENABLED and settings.RAG_ENABLED):
        return None
    embedder = get_rag_pipeline().embedder
    store = (
        InMemoryAnswerStore(similarity_threshold=settings.ANSWER_CACHE_THRESHOLD)
        if settings.ANSWER_CACHE_STORE == "memory"
        else PgVectorAnswerStore(
            _postgres_url(),
            table=settings.ANSWER_CACHE_TABLE,
            dimension=settings.RAG_EMBEDDING_DIM,
            similarity_threshold=settings.ANSWER_CACHE_THRESHOLD,
        )
    )
    return AnswerCache(
        store=store,
        embed=embedder.embed_query,
        # A new prompt, model, knowledge table or cache version never serves old answers.
        scope=scope_key(
            _SYSTEM_PROMPT_TEMPLATE,
            settings.DEFAULT_LLM_MODEL,
            settings.RAG_TABLE,
            settings.RAG_EMBEDDING_MODEL,
            settings.ANSWER_CACHE_VERSION,
        ),
        # Safe because the graph answers cache-eligible turns from a neutral prompt (no name, no memory).
        shared=True,
        similarity_threshold=settings.ANSWER_CACHE_THRESHOLD,
        ttl_s=settings.ANSWER_CACHE_TTL_S,
        max_stale_s=settings.ANSWER_CACHE_MAX_STALE_S,
    )


def get_answer_cache() -> Optional[AnswerCache]:
    """Return the process-wide answer cache (None when disabled), building it on first use."""
    global _answer_cache, _answer_cache_built
    if not _answer_cache_built:
        _answer_cache = build_answer_cache_from_settings()
        _answer_cache_built = True
    return _answer_cache


def set_answer_cache(cache: Optional[AnswerCache]) -> None:
    """Replace the process-wide answer cache (used by tests)."""
    global _answer_cache, _answer_cache_built
    _answer_cache, _answer_cache_built = cache, True


# ---------------------------------------------------------------- cost guard

_cost_guard: Optional[CostGuard] = None


def get_cost_guard() -> CostGuard:
    """Process-wide guard built from the COST_* settings."""
    global _cost_guard
    if _cost_guard is None:
        _cost_guard = CostGuard(
            cache_service,
            settings.COST_MONTHLY_BUDGET_USD,
            parse_prices(settings.COST_PRICES_JSON),
            (settings.COST_DEFAULT_INPUT_PER_1M, settings.COST_DEFAULT_OUTPUT_PER_1M),
        )
    return _cost_guard


def set_cost_guard(guard: Optional[CostGuard]) -> None:
    """Replace the process-wide guard (used by tests)."""
    global _cost_guard
    _cost_guard = guard


# ---------------------------------------------------------------- MCP


def build_toolkit_from_settings() -> Optional[MCPToolkit]:
    """Return a toolkit for ``MCP_CONFIG_FILE``, or None when MCP is not configured."""
    if not settings.MCP_CONFIG_FILE:
        return None
    allowlist = set(settings.MCP_TOOL_ALLOWLIST) or None
    return MCPToolkit(load_mcp_config(settings.MCP_CONFIG_FILE), allowlist=allowlist)
