"""Saathi knowledge base (RAG): curated plain-language guides in ``knowledge/*.md``.

Each ``## Heading`` section becomes one retrievable passage, indexed with the same
chunker, embeddings and ChromaDB store used for policies. The index is rebuilt
automatically when a guide changes (content hash), so editing a .md file is enough.
"""

from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path
from typing import Optional

from .embeddings import get_embedding_model
from .retriever import PolicyRetriever, RetrievedChunk
from .schemas import PageContent
from .text_chunker import chunk_pages
from .utils import CHROMA_PERSIST_DIR, ensure_dir, logger
from .vector_store import ChromaVectorStore

KB_ID = "saathi_kb"
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
_LOCK = threading.Lock()
_READY = False


def _sections() -> list[tuple[str, str, str]]:
    """(file name, section title, section text) for every ## section of every guide."""
    out = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc_title = (re.match(r"#\s+(.+)", text) or [None, path.stem])[1].strip()
        for block in re.split(r"(?m)^##\s+", text)[1:]:
            title, _, body = block.partition("\n")
            out.append((path.name, f"{doc_title}: {title.strip()}", f"{title.strip()}\n{body.strip()}"))
    return out


def _content_hash() -> str:
    h = hashlib.sha256()
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def ensure_kb(force: bool = False) -> int:
    """Index the guides if needed. Returns the number of passages."""
    global _READY
    with _LOCK:
        marker = ensure_dir(Path(CHROMA_PERSIST_DIR).parent) / "kb_hash.txt"
        digest = _content_hash()
        store = ChromaVectorStore()
        if not force and _READY and store.is_ingested(KB_ID):
            return store.count(KB_ID)
        if not force and marker.exists() and marker.read_text().strip() == digest and store.is_ingested(KB_ID):
            _READY = True
            return store.count(KB_ID)

        sections = _sections()
        if not sections:
            return 0
        if store.is_ingested(KB_ID):
            store.delete_policy(KB_ID)
        chunks = []
        for number, (fname, title, text) in enumerate(sections, 1):
            page = PageContent(page_number=number, page_text=text, source_filename=fname)
            for chunk in chunk_pages([page], chunk_size=900, overlap=120, source_filename=fname):
                chunk.section_name = title
                chunk.clause_reference = None
                chunks.append(chunk)
        embeddings = get_embedding_model().embed_texts([c.text for c in chunks])
        store.add_chunks(KB_ID, chunks, embeddings)
        marker.write_text(digest)
        _READY = True
        logger.info("Knowledge base indexed: %d passages from %d sections", len(chunks), len(sections))
        return len(chunks)


def search_kb(query: str, top_k: int = 4, min_relevance: float = 0.3) -> list[RetrievedChunk]:
    """Most relevant guide passages for a question (empty if none is close enough)."""
    ensure_kb()
    return PolicyRetriever().retrieve(KB_ID, query, top_k=top_k, min_relevance=min_relevance)
