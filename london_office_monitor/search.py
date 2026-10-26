"""Hybrid search over chunks (pgvector and full text, over each chunk and over its document's key, fused by reciprocal
rank), one hit per story with the nearby passages of its document; plus matching a document against the portfolio and
mandate, and recalling the analyst's past decisions."""
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import psycopg

from .embed import Embedder
from .facts import NotInData

# Reciprocal rank fusion constant from the original RRF paper; it damps the weight of top ranks.
RRF_K = 60
CANDIDATES_PER_RANKER = 50
MAX_PASSAGE_CHARS = 600
# A hit also carries its document's other passages, nearest first, up to four full-length passages of text: a short
# internal note comes whole (the bundled ones are at most 858 characters), a long one brings the sections around the hit.
NEARBY_PASSAGES_CHARS = 4 * MAX_PASSAGE_CHARS
SIMILARITY_PLACES = 3

# Four rankings fused by reciprocal rank: chunk vectors, chunk words, document-key vectors and document-key words. A
# news chunk is the feed summary alone, so a word that is only in the headline is found through the key.
SEARCH_SQL = """
WITH eligible AS (
  SELECT c.id AS chunk_id, c.document_id, c.embedding AS c_emb, c.tsv AS c_tsv
  FROM chunks c JOIN documents d ON d.id = c.document_id
  WHERE d.published_at <= %(until)s
    AND (%(since)s::timestamptz IS NULL OR d.published_at >= %(since)s)
    AND (%(kinds)s::text[] IS NULL OR d.source_kind = ANY(%(kinds)s))
),
by_vector AS (
  SELECT chunk_id, row_number() OVER (ORDER BY c_emb <=> %(q)s, chunk_id) AS r
  FROM eligible ORDER BY c_emb <=> %(q)s, chunk_id LIMIT %(n)s
),
by_text AS (
  SELECT chunk_id, row_number() OVER (ORDER BY ts_rank_cd(c_tsv, query) DESC, chunk_id) AS r
  FROM eligible, websearch_to_tsquery('english', %(text)s) AS query
  WHERE c_tsv @@ query ORDER BY ts_rank_cd(c_tsv, query) DESC, chunk_id LIMIT %(n)s
),
by_key AS (
  SELECT d.id AS document_id, row_number() OVER (ORDER BY d.embedding <=> %(q)s, d.id) AS r
  FROM documents d
  WHERE d.published_at <= %(until)s
    AND (%(since)s::timestamptz IS NULL OR d.published_at >= %(since)s)
    AND (%(kinds)s::text[] IS NULL OR d.source_kind = ANY(%(kinds)s))
  ORDER BY d.embedding <=> %(q)s, d.id LIMIT %(n)s
),
by_key_text AS (
  SELECT d.id AS document_id, row_number() OVER (ORDER BY ts_rank_cd(d.tsv, query) DESC, d.id) AS r
  FROM documents d, websearch_to_tsquery('english', %(text)s) AS query
  WHERE d.tsv @@ query AND d.published_at <= %(until)s
    AND (%(since)s::timestamptz IS NULL OR d.published_at >= %(since)s)
    AND (%(kinds)s::text[] IS NULL OR d.source_kind = ANY(%(kinds)s))
  ORDER BY ts_rank_cd(d.tsv, query) DESC, d.id LIMIT %(n)s
)
SELECT e.chunk_id, e.document_id,
       coalesce(1.0 / (%(k)s + v.r), 0) + coalesce(1.0 / (%(k)s + t.r), 0) + coalesce(1.0 / (%(k)s + dk.r), 0)
       + coalesce(1.0 / (%(k)s + kt.r), 0) AS score
FROM eligible e
LEFT JOIN by_vector v ON v.chunk_id = e.chunk_id
LEFT JOIN by_text t ON t.chunk_id = e.chunk_id
LEFT JOIN by_key dk ON dk.document_id = e.document_id
LEFT JOIN by_key_text kt ON kt.document_id = e.document_id
WHERE v.r IS NOT NULL OR t.r IS NOT NULL OR dk.r IS NOT NULL OR kt.r IS NOT NULL
ORDER BY score DESC, e.chunk_id
"""

HIT_SQL = """
SELECT c.id, c.document_id, c.chunk_index, d.title, c.text, c.heading_path, c.is_table, d.publisher, d.published_at,
       d.fetched_at, d.url, d.story_id, d.synthetic, d.source_kind
FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = ANY(%s)
"""


@dataclass(frozen=True)
class NearbyPassage:
    citation: str
    heading_path: list[str]
    quoted_passage: str
    is_table: bool


@dataclass(frozen=True)
class SearchHit:
    doc_id: int
    chunk_index: int
    citation: str
    quoted_title: str
    quoted_passage: str
    heading_path: list[str]
    is_table: bool
    publisher: str
    published_at: datetime
    fetched_at: datetime
    url: str
    story_id: int
    also_reported_by: list[str]
    synthetic: bool
    source_kind: str
    score: float
    nearby_passages: list[NearbyPassage]


def _vector(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _nearby_passages(conn: psycopg.Connection, doc_id: int, chunk_index: int) -> list[NearbyPassage]:
    """The document's other passages, nearest the hit first (the earlier one at equal distance) while their quoted text
    fits NEARBY_PASSAGES_CHARS, returned in reading order. Each keeps its own citation, so an answer cites the passage
    it uses."""
    rows = conn.execute("SELECT chunk_index, heading_path, text, is_table FROM chunks WHERE document_id = %s"
                        " AND chunk_index <> %s", (doc_id, chunk_index)).fetchall()
    chosen: list[tuple[int, NearbyPassage]] = []
    used = 0
    for index, heading_path, text, is_table in sorted(rows, key=lambda r: (abs(r[0] - chunk_index), r[0])):
        quoted = text[:MAX_PASSAGE_CHARS]
        if used + len(quoted) > NEARBY_PASSAGES_CHARS:
            break
        used += len(quoted)
        chosen.append((index, NearbyPassage(citation=f"D{doc_id}#{index}", heading_path=list(heading_path),
                                            quoted_passage=quoted, is_table=is_table)))
    return [passage for _, passage in sorted(chosen, key=lambda item: item[0])]


def other_outlets(conn: psycopg.Connection, story_id: int, publisher: str, until: datetime | None = None) -> list[str]:
    """Publishers of the same story other than this one, in the order they published it."""
    rows = conn.execute("SELECT publisher, min(published_at) FROM documents WHERE story_id = %s AND publisher <> %s"
                        " AND (%s::timestamptz IS NULL OR published_at <= %s) GROUP BY publisher ORDER BY 2, 1",
                        (story_id, publisher, until, until))
    return [r[0] for r in rows]


def search_documents(conn: psycopg.Connection, embedder: Embedder, query: str, *, until: datetime,
                     since: datetime | None = None, source_types: list[str] | None = None,
                     k: int = 8) -> list[SearchHit]:
    """The k best passages, at most one per story, published no later than `until`."""
    params = {"until": until, "since": since, "kinds": list(source_types) if source_types else None,
              "q": _vector(embedder.embed_query(query)), "text": query, "n": CANDIDATES_PER_RANKER, "k": RRF_K}
    ranked = conn.execute(SEARCH_SQL, params).fetchall()
    best_per_document: dict[int, tuple[int, float]] = {}
    for chunk_id, document_id, score in ranked:
        best_per_document.setdefault(document_id, (chunk_id, float(score)))
    if not best_per_document:
        return []
    details = {row[0]: row for row in conn.execute(HIT_SQL, ([c for c, _ in best_per_document.values()],))}
    hits: list[SearchHit] = []
    stories: set[int] = set()
    for document_id, (chunk_id, score) in sorted(best_per_document.items(), key=lambda item: (-item[1][1], item[0])):
        row = details[chunk_id]
        story_id = row[11]
        if story_id in stories:
            continue
        stories.add(story_id)
        hits.append(SearchHit(doc_id=row[1], chunk_index=row[2], citation=f"D{row[1]}#{row[2]}", quoted_title=row[3],
                              quoted_passage=row[4][:MAX_PASSAGE_CHARS], heading_path=list(row[5]), is_table=row[6],
                              publisher=row[7], published_at=row[8], fetched_at=row[9], url=row[10],
                              story_id=story_id, also_reported_by=other_outlets(conn, story_id, row[7], until),
                              synthetic=row[12], source_kind=row[13], score=round(score, 6),
                              nearby_passages=_nearby_passages(conn, row[1], row[2])))
        if len(hits) == k:
            break
    return hits


def match_to_mandate(conn: psycopg.Connection, doc_id: int) -> list[dict]:
    """How close a document is to each portfolio asset and mandate statement, closest first."""
    if conn.execute("SELECT 1 FROM documents WHERE id = %s", (doc_id,)).fetchone() is None:
        raise NotInData(f"No document D{doc_id}.")
    rows = conn.execute(
        "SELECT a.name, 'asset', 1 - (a.embedding <=> d.embedding) FROM portfolio_assets a, documents d WHERE d.id = %s"
        " UNION ALL SELECT m.text, 'mandate', 1 - (m.embedding <=> d.embedding) FROM mandate_statements m, documents d"
        " WHERE d.id = %s ORDER BY 3 DESC, 1", (doc_id, doc_id))
    return [{"target": r[0], "kind": r[1], "similarity": round(float(r[2]), SIMILARITY_PLACES)} for r in rows]


def similar_past_decisions(conn: psycopg.Connection, embedder: Embedder, text: str, k: int = 3) -> list[dict]:
    """The analyst's closest earlier decisions on opportunity cards, with the reasons given."""
    vector = _vector(embedder.embed_documents([text])[0])
    rows = conn.execute(
        "SELECT id, card->>'title', status, decision_reason, 1 - (embedding <=> %s) FROM opportunities"
        " WHERE status IN ('approved', 'rejected') AND embedding IS NOT NULL ORDER BY embedding <=> %s, id LIMIT %s",
        (vector, vector, k))
    return [{"card_id": r[0], "title": r[1], "decision": r[2], "reason": r[3],
             "similarity": round(float(r[4]), SIMILARITY_PLACES)} for r in rows]
