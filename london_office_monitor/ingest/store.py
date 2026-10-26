"""Store documents with their chunks: exact duplicates are skipped, near-duplicates join an existing story."""
import hashlib
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

from ..chunking import ChunkIn, chunk_text, contextual_header, document_key_text
from ..clock import now
from ..config import Settings
from ..embed import Embedder
from .feeds import DocumentIn, bounded


@dataclass(frozen=True)
class StoreResult:
    new: int
    duplicates: int
    joined_story: int


def content_hash(doc: DocumentIn) -> str:
    return hashlib.sha256(f"{doc.url}\n{doc.title}".encode()).hexdigest()


def _vector(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def plan_chunks(doc: DocumentIn, settings: Settings) -> list[ChunkIn]:
    """A document's chunks: its summary (or its title when there is none) split along its structure."""
    return chunk_text(doc.summary or doc.title, fmt=doc.text_format, max_tokens=settings.chunk_max_tokens)


def chunk_embedding_text(doc: DocumentIn, chunk: ChunkIn) -> str:
    """What a chunk is embedded as: its contextual header, then its text."""
    header = contextual_header(doc.publisher, doc.title, doc.subtitle, chunk.heading_path, doc.published_at)
    return f"{header}\n{chunk.text}"


def store_documents(conn: psycopg.Connection, docs: list[DocumentIn], embedder: Embedder,
                    settings: Settings) -> StoreResult:
    """Insert the new documents and their chunks in one transaction. A document's story is the closest stored
    document published within the window when the two are at least as similar as the threshold, else its own id.
    Ids come from the sequence before the insert, so no UPDATE is needed (the MCP role may only insert)."""
    fresh: list[tuple[DocumentIn, str]] = []
    seen: set[str] = set()
    duplicates = 0
    for doc in docs:
        digest = content_hash(doc)
        if digest in seen or conn.execute("SELECT 1 FROM documents WHERE content_hash = %s", (digest,)).fetchone():
            duplicates += 1
            continue
        seen.add(digest)
        fresh.append((doc, digest))
    if not fresh:
        return StoreResult(new=0, duplicates=duplicates, joined_story=0)

    plans = [plan_chunks(doc, settings) for doc, _ in fresh]
    key_vectors = embedder.embed_documents([document_key_text(doc) for doc, _ in fresh])
    chunk_vectors = iter(embedder.embed_documents(
        [chunk_embedding_text(doc, c) for (doc, _), chunks in zip(fresh, plans, strict=True) for c in chunks]))
    window = timedelta(days=settings.dedup_window_days)
    fetched_at = now(settings)
    joined = 0
    with conn.transaction():
        for (doc, digest), key_vector, chunks in zip(fresh, key_vectors, plans, strict=True):
            vector = _vector(key_vector)
            doc_id = conn.execute("SELECT nextval('documents_id_seq')").fetchone()[0]
            nearest = conn.execute(
                "SELECT story_id, 1 - (embedding <=> %s) FROM documents WHERE published_at BETWEEN %s AND %s"
                " ORDER BY embedding <=> %s LIMIT 1",
                (vector, doc.published_at - window, doc.published_at + window, vector)).fetchone()
            story_id = nearest[0] if nearest and nearest[1] >= settings.dedup_similarity else doc_id
            joined += story_id != doc_id
            # A feed summary is bounded; an internal document is stored whole, as it was chunked (read_internal
            # refuses one longer than the column allows).
            summary = doc.summary if doc.source_kind == "internal" else bounded(doc.summary)
            conn.execute(
                "INSERT INTO documents (id, source_kind, publisher, title, subtitle, summary, url, published_at,"
                " fetched_at, content_hash, story_id, synthetic, fixture_role, meta, embedding)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (doc_id, doc.source_kind, doc.publisher, doc.title, doc.subtitle, summary, doc.url,
                 doc.published_at, fetched_at, digest, story_id, doc.synthetic, doc.fixture_role, Jsonb(doc.meta),
                 vector))
            for chunk in chunks:
                conn.execute(
                    "INSERT INTO chunks (document_id, chunk_index, heading_path, text, char_start, char_end, is_table,"
                    " embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (doc_id, chunk.chunk_index, list(chunk.heading_path), chunk.text, chunk.char_start,
                     chunk.char_end, chunk.is_table, _vector(next(chunk_vectors))))
    return StoreResult(new=len(fresh), duplicates=duplicates, joined_story=joined)
