"""RAG layer tests. Everything runs offline with the hashing embedder and the in-memory store."""

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver

from app.core.langgraph.graph import LangGraphAgent
from app.core.langgraph.tools.knowledge_search import knowledge_search
from app.services.rag import (
    Document,
    HashingEmbedder,
    InMemoryVectorStore,
    RAGPipeline,
    format_context,
    set_rag_pipeline,
)
from app.services.rag.chunking import (
    chunk_documents,
    load_documents,
)
from app.services.rag.contracts import (
    Chunk,
    SearchResult,
)
from app.services.rag.ranking import (
    BM25Index,
    reciprocal_rank_fusion,
    tokenize,
)

DOCS = [
    Document(
        id="policy/refund.md",
        title="Refund policy",
        source="refund.md",
        content="Customers can request a refund within 30 days of purchase. Refunds go to the original payment method.",
    ),
    Document(
        id="policy/warranty.md",
        title="Warranty",
        source="warranty.md",
        content="All laptops carry a 24 month warranty covering manufacturing defects but not accidental damage.",
    ),
    Document(
        id="news/launch.md",
        title="Store launch",
        source="launch.md",
        url="https://example.com/launch",
        content="The new Hanoi store opens on Saturday with discounts on headphones.",
    ),
]


class CountingEmbedder(HashingEmbedder):
    """Hashing embedder that records how many passages it embedded."""

    def __init__(self):
        super().__init__(dimension=128)
        self.embedded = 0

    async def embed_documents(self, texts):
        self.embedded += len(texts)
        return await super().embed_documents(texts)


def _chunk(cid: str) -> Chunk:
    return Chunk(id=cid, document_id=cid, chunk_index=0, content=cid, title=cid, source=cid)


@pytest.fixture
def pipeline():
    return RAGPipeline(embedder=CountingEmbedder(), store=InMemoryVectorStore(), score_threshold=0.2)


def test_chunk_ids_are_stable_and_indexed():
    long_doc = Document(id="a.md", title="A", source="a.md", content=("word " * 300).strip())
    first = chunk_documents([long_doc], chunk_size=200, chunk_overlap=20)
    second = chunk_documents([long_doc], chunk_size=200, chunk_overlap=20)

    assert len(first) > 1
    assert [c.id for c in first] == [c.id for c in second]
    assert [c.chunk_index for c in first] == list(range(len(first)))
    assert first[0].id == "a.md::chunk-0"


def test_load_documents_reads_title_and_url(tmp_path):
    (tmp_path / "legal").mkdir()
    (tmp_path / "legal" / "rules.md").write_text(
        "---\ntitle: Student rules\nurl: https://uni.example/rules\n---\nBody text.", encoding="utf-8"
    )
    (tmp_path / "broken.pdf").write_bytes(b"%PDF not really")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    docs = load_documents(tmp_path)

    # The broken PDF is skipped with a warning and unsupported files are ignored.
    assert [d.id for d in docs] == ["legal/rules.md"]
    assert docs[0].title == "Student rules"
    assert docs[0].url == "https://uni.example/rules"


def test_rrf_rewards_agreement_between_rankings():
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    dense = [SearchResult(chunk=a, score=0.9, method="dense"), SearchResult(chunk=b, score=0.8, method="dense")]
    lexical = [SearchResult(chunk=b, score=7.0, method="bm25"), SearchResult(chunk=c, score=3.0, method="bm25")]

    fused = reciprocal_rank_fusion([dense, lexical], top_k=3)

    assert [r.chunk.id for r in fused] == ["b", "a", "c"]
    assert all(r.method == "hybrid" for r in fused)


def test_bm25_ignores_non_matching_chunks():
    index = BM25Index([_chunk("refund window"), _chunk("warranty months"), _chunk("store opening")])

    results = index.search("refund", top_k=5)

    assert [r.chunk.id for r in results] == ["refund window"]


async def test_reingest_skips_unchanged_chunks(pipeline):
    first = await pipeline.ingest(DOCS)
    second = await pipeline.ingest(DOCS)

    assert first.embedded == first.chunks == 3
    assert second.embedded == 0 and second.reused == 3
    assert pipeline.embedder.embedded == 3


async def test_reingest_drops_chunks_a_shorter_document_no_longer_has():
    pipeline = RAGPipeline(embedder=HashingEmbedder(64), store=InMemoryVectorStore(), chunk_size=60, chunk_overlap=0)
    long_doc = Document(
        id="faq.md", title="FAQ", source="faq.md", content=" ".join(f"Old fact {i} about returns." for i in range(12))
    )
    await pipeline.ingest([long_doc])

    report = await pipeline.ingest([long_doc.model_copy(update={"content": "Returns are free."})])
    chunks = await pipeline.store.all_chunks()
    result = await pipeline.retrieve("old fact 7")

    assert [c.id for c in chunks] == ["faq.md::chunk-0"]
    assert report.removed_chunks > 0 and report.changed_documents == ["faq.md"]
    assert all("Old fact" not in r.chunk.content for r in result.results)


async def test_prune_removes_documents_missing_from_the_corpus(pipeline):
    await pipeline.ingest(DOCS)

    partial = await pipeline.ingest(DOCS[:2])  # a partial batch never deletes other documents
    assert len(await pipeline.store.all_chunks()) == 3 and partial.removed_documents == []

    report = await pipeline.ingest(DOCS[:2], prune_missing=True)
    remaining = {c.document_id for c in await pipeline.store.all_chunks()}

    assert report.removed_documents == ["news/launch.md"]
    assert "news/launch.md" in report.changed_documents  # so cached answers built on it are invalidated
    assert remaining == {"policy/refund.md", "policy/warranty.md"}


async def test_retrieve_finds_relevant_chunk_and_flags_confidence(pipeline):
    await pipeline.ingest(DOCS)

    hit = await pipeline.retrieve("how many days do I have to request a refund", top_k=2)
    miss = await pipeline.retrieve("quantum chromodynamics lattice", top_k=2)

    assert hit.results[0].chunk.document_id == "policy/refund.md"
    assert not hit.low_confidence
    assert miss.low_confidence


async def test_retrieve_survives_embedding_outage(pipeline):
    await pipeline.ingest(DOCS)

    async def broken(_):
        raise RuntimeError("provider down")

    pipeline.embedder.embed_query = broken
    result = await pipeline.retrieve("warranty laptops")

    assert result.results[0].chunk.document_id == "policy/warranty.md"
    assert result.low_confidence  # no dense score, so the gate stays closed


async def test_format_context_numbers_sources_in_rank_order(pipeline):
    await pipeline.ingest(DOCS)

    context = format_context(await pipeline.retrieve("Hanoi store discounts", top_k=1))

    assert "[1] Store launch (https://example.com/launch)" in context
    assert "Cite every claim" in context


async def test_agent_calls_knowledge_search_tool(pipeline, monkeypatch):
    await pipeline.ingest(DOCS)
    set_rag_pipeline(pipeline)
    try:
        agent = LangGraphAgent()
        agent.tools_by_name = {"knowledge_search": knowledge_search}
        replies = [
            AIMessage(
                content="",
                tool_calls=[{"name": "knowledge_search", "args": {"query": "refund days"}, "id": "c1"}],
            ),
            AIMessage(content="You have 30 days [1]."),
        ]

        async def scripted(messages, *args, **kwargs):
            return replies.pop(0)

        monkeypatch.setattr(agent.llm_service, "call", scripted)
        graph = await agent.create_graph(checkpointer=InMemorySaver())
        state = await graph.ainvoke(
            {"messages": [HumanMessage(content="How long is the refund window?")]},
            config={"configurable": {"thread_id": "rag"}},
        )
    finally:
        set_rag_pipeline(None)

    tool_output = next(m for m in state["messages"] if isinstance(m, ToolMessage)).content
    assert "[1] Refund policy" in tool_output
    assert state["messages"][-1].content == "You have 30 days [1]."


def test_html_is_converted_to_markdown(tmp_path):
    (tmp_path / "page.html").write_text(
        "<html><body><h1>Opening hours</h1><p>Open <b>9am to 5pm</b> on weekdays.</p></body></html>",
        encoding="utf-8",
    )

    [doc] = load_documents(tmp_path)

    assert doc.title == "Opening hours"
    assert "9am to 5pm" in doc.content


def test_vietnamese_compounds_become_single_tokens():
    assert "học_phí" in tokenize("Sinh viên phải đóng học phí")
    assert tokenize("Sinh viên phải đóng học phí", mode="simple") == ["sinh", "viên", "phải", "đóng", "học", "phí"]
    assert tokenize("refund within 30 days") == ["refund", "within", "30", "days"]


def test_vietnamese_bm25_does_not_match_split_syllables():
    tuition = _chunk("tuition")
    tuition = tuition.model_copy(update={"content": "Sinh viên đóng học phí trước ngày 15 tháng 9."})
    parking = _chunk("parking")
    parking = parking.model_copy(update={"content": "Môn học thể dục được miễn phí gửi xe."})

    segmented = BM25Index([tuition, parking], mode="auto").search("học phí", top_k=5)
    naive = BM25Index([tuition, parking], mode="simple").search("học phí", top_k=5)

    assert [r.chunk.id for r in segmented] == ["tuition"]
    assert {r.chunk.id for r in naive} == {"tuition", "parking"}


def test_bm25_rebuild_reuses_tokens_of_unchanged_chunks(monkeypatch):
    import app.services.rag.ranking as ranking

    calls = []
    real = ranking.tokenize
    monkeypatch.setattr(ranking, "tokenize", lambda text, mode="auto": calls.append(text) or real(text, mode))

    first = BM25Index([_chunk("a"), _chunk("b")])
    calls.clear()
    BM25Index([_chunk("a"), _chunk("b"), _chunk("c")], reuse=first)

    assert calls == ["c"]


class ReversingReranker:
    async def rerank(self, query, candidates, top_k):
        return [c.model_copy(update={"method": "rerank"}) for c in reversed(candidates)][:top_k]


class BrokenReranker:
    async def rerank(self, query, candidates, top_k):
        raise RuntimeError("model failed to load")


async def test_reranker_reorders_fused_candidates(pipeline):
    await pipeline.ingest(DOCS)
    baseline = await pipeline.retrieve("refund warranty store", top_k=3)

    pipeline.reranker = ReversingReranker()
    reranked = await pipeline.retrieve("refund warranty store", top_k=3)

    assert all(r.method == "rerank" for r in reranked.results)
    assert reranked.results[0].chunk.id != baseline.results[0].chunk.id


async def test_failing_reranker_falls_back_to_fused_order(pipeline):
    await pipeline.ingest(DOCS)
    pipeline.reranker = BrokenReranker()

    result = await pipeline.retrieve("refund policy", top_k=2)

    assert len(result.results) == 2
    assert result.results[0].method == "hybrid"


@pytest.mark.slow
async def test_flashrank_reranker_puts_answer_first():
    pytest.importorskip("rerankers")
    from app.services.rag import RerankersReranker

    try:
        reranker = RerankersReranker("flashrank")
    except Exception as e:  # first run downloads a small ONNX model
        pytest.skip(f"flashrank model unavailable: {e}")

    candidates = [
        SearchResult(chunk=c.model_copy(update={"content": text}), score=0.0, method="hybrid")
        for c, text in [
            (_chunk("store"), "The new Hanoi store opens on Saturday."),
            (_chunk("refund"), "Customers can request a refund within 30 days of purchase."),
            (_chunk("warranty"), "Laptops carry a 24 month warranty."),
        ]
    ]

    ranked = await reranker.rerank("How many days do I have to ask for a refund?", candidates, top_k=2)

    assert [r.chunk.id for r in ranked][0] == "refund"
    assert len(ranked) == 2


async def test_tokens_stored_at_ingest_spare_query_time_tokenization(monkeypatch):
    import app.services.rag.ranking as ranking

    store = InMemoryVectorStore()
    await RAGPipeline(embedder=HashingEmbedder(64), store=store).ingest(DOCS)

    calls = []
    real = ranking.tokenize
    monkeypatch.setattr(ranking, "tokenize", lambda text, mode="auto": calls.append(text) or real(text, mode))
    # A fresh pipeline over the same store stands in for a newly started server process.
    result = await RAGPipeline(embedder=HashingEmbedder(64), store=store).retrieve("laptop warranty")

    assert calls == ["laptop warranty"]  # only the query was tokenized
    assert result.results[0].chunk.document_id == "policy/warranty.md"


async def test_answer_citing_a_missing_source_is_flagged(pipeline, monkeypatch):
    from prometheus_client import REGISTRY

    await pipeline.ingest(DOCS)
    set_rag_pipeline(pipeline)
    before = REGISTRY.get_sample_value("rag_citation_issues_total", {"kind": "out_of_range"}) or 0.0
    try:
        agent = LangGraphAgent()
        agent.tools_by_name = {"knowledge_search": knowledge_search}
        replies = [
            AIMessage(content="", tool_calls=[{"name": "knowledge_search", "args": {"query": "refund"}, "id": "c1"}]),
            AIMessage(content="Refunds take 30 days [7]."),  # only a handful of sources were returned
        ]

        async def scripted(messages, *args, **kwargs):
            return replies.pop(0)

        monkeypatch.setattr(agent.llm_service, "call", scripted)
        graph = await agent.create_graph(checkpointer=InMemorySaver())
        await graph.ainvoke({"messages": [HumanMessage(content="refund?")]}, {"configurable": {"thread_id": "cite"}})
    finally:
        set_rag_pipeline(None)

    assert REGISTRY.get_sample_value("rag_citation_issues_total", {"kind": "out_of_range"}) == before + 1
