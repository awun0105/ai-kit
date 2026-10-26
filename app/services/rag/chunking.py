"""Document loading and chunking.

Ported from D08 ``task4_chunking_indexing.py``: recursive character splitting
with stable chunk ids, so re-ingesting the same corpus upserts instead of
duplicating. PDF, DOCX and HTML are converted to Markdown with Microsoft
MarkItDown, as in D08 ``task3_convert_markdown.py``.
"""

import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown

from app.core.logging import logger
from app.services.rag.contracts import (
    Chunk,
    Document,
)

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
TEXT_SUFFIXES = {".md", ".txt"}
CONVERTED_SUFFIXES = {".pdf", ".docx", ".html", ".htm"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | CONVERTED_SUFFIXES
_BINARY_SIGNATURES = ("%PDF", "PK\x03\x04")  # PDF header, ZIP header (DOCX)

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)[ \t]*$", re.MULTILINE)
_FRONTMATTER_TITLE = re.compile(r"^title:\s*[\"']?(.*?)[\"']?\s*$", re.MULTILINE)
_FRONTMATTER_URL = re.compile(r"^url:\s*[\"']?(https?://[^\s\"']+)[\"']?\s*$", re.MULTILINE)
_SOURCE_LINE = re.compile(r"^\s*\*\*Source:\*\*\s*(?P<url>\S+)[ \t]*$", re.MULTILINE)


def _title(content: str, fallback: str) -> str:
    match = _FRONTMATTER_TITLE.search(content)
    if match and match.group(1).strip():
        return match.group(1).strip()
    match = _HEADING.search(content)
    return match.group("title").strip() if match else fallback


def _url(content: str) -> str | None:
    match = _FRONTMATTER_URL.search(content) or _SOURCE_LINE.search(content)
    return match.group(1).strip() if match else None


def read_as_markdown(path: Path) -> str:
    """Return the file as Markdown: text files as-is, PDF/DOCX/HTML via MarkItDown."""
    if path.suffix.lower() in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8")
    text = MarkItDown(enable_plugins=False).convert(str(path)).text_content
    # MarkItDown falls back to reading unparseable files as plain text, which
    # would index raw PDF/ZIP bytes. Treat that as a failed conversion.
    if text.lstrip().startswith(_BINARY_SIGNATURES):
        raise ValueError(f"could not convert {path.name}; output is the raw file")
    return text


def load_documents(root: Path) -> list[Document]:
    """Read every supported file under ``root`` into a Document.

    Files that fail to convert are skipped with a warning rather than aborting the run.

    The id is the path relative to ``root``, which keeps it stable between runs.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        try:
            content = read_as_markdown(path).strip()
        except Exception as e:
            logger.warning("rag_document_conversion_failed", path=str(path), error=str(e))
            continue
        if not content:
            continue
        relative = path.relative_to(root).as_posix()
        documents.append(
            Document(
                id=relative,
                content=content,
                title=_title(content, path.stem),
                source=path.name,
                url=_url(content),
                metadata={"folder": path.parent.relative_to(root).as_posix()},
            )
        )
    return documents


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split documents into chunks; ``chunk_index`` counts only non-empty chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )
    chunks: list[Chunk] = []
    for document in documents:
        index = 0
        for text in splitter.split_text(document.content):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    id=f"{document.id}::chunk-{index}",
                    document_id=document.id,
                    chunk_index=index,
                    content=text,
                    title=document.title,
                    source=document.source,
                    url=document.url,
                    metadata=document.metadata,
                )
            )
            index += 1
    return chunks
