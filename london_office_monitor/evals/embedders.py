"""The embedding comparison behind `lom eval-embedders`: each candidate model embeds the same corpus (the fixture
snapshot and the internal documents, as `lom setup` stores them) and answers the same questions in English and in
Chinese (evals/retrieval_cases.yaml). Documents are ranked by search_documents' two vector rankings, chunks and
document keys, fused by reciprocal rank; its full-text ranking is left out because it is the same for every model."""
import functools
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..chunking import document_key_text
from ..config import Settings
from ..embed import QUERY_INSTRUCTION
from ..ingest.feeds import DocumentIn
from ..ingest.store import chunk_embedding_text, content_hash, plan_chunks
from ..paths import PROJECT_ROOT
from ..search import CANDIDATES_PER_RANKER, RRF_K

RETRIEVAL_CASES_FILE = PROJECT_ROOT / "evals" / "retrieval_cases.yaml"
# The folder lom setup downloads the production model into (the default of Settings.embedding_cache_dir).
MODEL_CACHE_DIR = PROJECT_ROOT / ".cache" / "fastembed"
EMBED_BATCH_SIZE = 32
# Hit@5 counts a question as answered when an answer ranks in the top 5, as an agent reading five hits would see it;
# the reciprocal rank counts answers down to rank 10.
HIT_DEPTH = 5
MRR_DEPTH = 10
MAX_ERROR_CHARS = 300
Language = Literal["en", "zh"]
LANGUAGES: tuple[Language, ...] = ("en", "zh")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalTopic(StrictModel):
    """One question asked in English and in Chinese, with the URL of every document that answers it."""
    id: str = Field(pattern=r"^[a-z0-9_]{1,40}$")
    relevant: list[str] = Field(min_length=1)
    en: str = Field(min_length=1, max_length=300)
    zh: str = Field(min_length=1, max_length=300)


class RetrievalCases(StrictModel):
    topics: list[RetrievalTopic] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self):
        ids = [t.id for t in self.topics]
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        if repeated:
            raise ValueError(f"duplicate topic ids: {', '.join(repeated)}")
        return self


def load_retrieval_cases(path: Path) -> RetrievalCases:
    """The questions in a YAML file, with every problem reported together."""
    try:
        return RetrievalCases.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as e:
        raise ValueError(f"{path}: {e}") from e


@dataclass(frozen=True)
class EmbedderSpec:
    """A candidate model and the markers its authors ask for on queries and documents (fastembed adds none)."""
    name: str
    dim: int
    query_prefix: str
    document_prefix: str
    licence: str


EMBEDDER_SPECS: dict[str, EmbedderSpec] = {spec.name: spec for spec in (
    EmbedderSpec("Qwen/Qwen3-Embedding-0.6B-Q", 1024, QUERY_INSTRUCTION, "", "Apache 2.0"),
    EmbedderSpec("google/embeddinggemma-300m", 768, "task: search result | query: ", "title: none | text: ",
                 "Gemma terms of use"),
    EmbedderSpec("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384, "", "", "Apache 2.0"),
)}


@dataclass
class Corpus:
    """Every stored document's key text and URL, and every chunk's text with the index of its document."""
    urls: list[str]
    key_texts: list[str]
    chunk_texts: list[str]
    chunk_document: list[int]


def build_corpus(docs: list[DocumentIn], settings: Settings) -> Corpus:
    """The texts lom setup embeds for these documents, exact duplicates dropped as store_documents drops them."""
    seen: set[str] = set()
    unique = []
    for doc in docs:
        digest = content_hash(doc)
        if digest not in seen:
            seen.add(digest)
            unique.append(doc)
    chunk_texts, chunk_document = [], []
    for index, doc in enumerate(unique):
        for chunk in plan_chunks(doc, settings):
            chunk_texts.append(chunk_embedding_text(doc, chunk))
            chunk_document.append(index)
    return Corpus(urls=[d.url for d in unique], key_texts=[document_key_text(d) for d in unique],
                  chunk_texts=chunk_texts, chunk_document=chunk_document)


def _ranks(similarities: np.ndarray) -> dict[int, int]:
    """1-based ranks of the closest CANDIDATES_PER_RANKER rows, ties broken by row order as the SQL's ORDER BY does."""
    order = np.lexsort((np.arange(len(similarities)), -similarities))[:CANDIDATES_PER_RANKER]
    return {int(row): rank for rank, row in enumerate(order, start=1)}


def rank_documents(query: np.ndarray, keys: np.ndarray, chunks: np.ndarray, chunk_document: list[int]) -> list[int]:
    """Documents in the order search_documents' vector rankings put them: a chunk scores 1/(k + its rank) plus
    1/(k + its document key's rank), and a document scores by its best chunk."""
    chunk_rank, key_rank = _ranks(chunks @ query), _ranks(keys @ query)
    best: dict[int, float] = {}
    for row, document in enumerate(chunk_document):
        if row not in chunk_rank and document not in key_rank:
            continue
        score = (1 / (RRF_K + chunk_rank[row]) if row in chunk_rank else 0.0) \
            + (1 / (RRF_K + key_rank[document]) if document in key_rank else 0.0)
        best[document] = max(best.get(document, 0.0), score)
    return sorted(best, key=lambda document: (-best[document], document))


EmbedTexts = Callable[[EmbedderSpec, list[str]], np.ndarray]


@functools.cache
def _fastembed_model(name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(name, cache_dir=str(MODEL_CACHE_DIR))


def fastembed_texts(spec: EmbedderSpec, texts: list[str]) -> np.ndarray:
    """The texts' vectors from the local model, one row each."""
    vectors = np.array([np.asarray(v, dtype=np.float32)
                        for v in _fastembed_model(spec.name).embed(texts, batch_size=EMBED_BATCH_SIZE)])
    if vectors.ndim != 2 or vectors.shape[1] != spec.dim:
        raise ValueError(f"{spec.name} returned vectors of shape {vectors.shape}, expected {spec.dim} dimensions")
    return vectors


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


@dataclass
class QuestionResult:
    topic_id: str
    language: Language
    # The 1-based rank of the first document that answers the question; None when no answer is ranked at all.
    rank: int | None
    top_url: str | None


@dataclass
class EmbedderResult:
    model: str
    dim: int
    licence: str
    questions: list[QuestionResult]
    corpus_seconds: float
    query_seconds: float
    error: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


def compare_embedders(specs: list[EmbedderSpec], cases: RetrievalCases, corpus: Corpus,
                      embed: EmbedTexts | None = None) -> list[EmbedderResult]:
    """Each model embeds the corpus and the questions with its own markers, then ranks the documents for each
    question. A model that fails to load or embed is reported, and the others still run."""
    embed = embed or fastembed_texts
    questions = [(topic, language, getattr(topic, language)) for topic in cases.topics for language in LANGUAGES]
    results = []
    for spec in specs:
        try:
            started = time.monotonic()
            keys = _unit_rows(embed(spec, [spec.document_prefix + text for text in corpus.key_texts]))
            chunks = _unit_rows(embed(spec, [spec.document_prefix + text for text in corpus.chunk_texts]))
            corpus_seconds = time.monotonic() - started
            started = time.monotonic()
            queries = _unit_rows(embed(spec, [spec.query_prefix + text for _, _, text in questions]))
            query_seconds = time.monotonic() - started
        except Exception as e:  # noqa: BLE001  a model that cannot load or embed is this candidate's result
            results.append(EmbedderResult(spec.name, spec.dim, spec.licence, [], 0.0, 0.0,
                                          error=f"{type(e).__name__}: {e}"[:MAX_ERROR_CHARS]))
            continue
        answers = []
        for (topic, language, _), vector in zip(questions, queries, strict=True):
            ranking = rank_documents(vector, keys, chunks, corpus.chunk_document)
            relevant = set(topic.relevant)
            rank = next((i for i, document in enumerate(ranking, start=1) if corpus.urls[document] in relevant), None)
            answers.append(QuestionResult(topic.id, language, rank, corpus.urls[ranking[0]] if ranking else None))
        results.append(EmbedderResult(spec.name, spec.dim, spec.licence, answers, corpus_seconds, query_seconds))
    return results


@dataclass
class RetrievalStats:
    questions: int
    hit_at_1: float
    hit_at_5: float
    mrr: float


def _stats(questions: list[QuestionResult]) -> RetrievalStats:
    count = len(questions)
    if not count:
        return RetrievalStats(0, 0.0, 0.0, 0.0)
    return RetrievalStats(
        questions=count, hit_at_1=sum(q.rank == 1 for q in questions) / count,
        hit_at_5=sum(q.rank is not None and q.rank <= HIT_DEPTH for q in questions) / count,
        mrr=sum(1 / q.rank for q in questions if q.rank is not None and q.rank <= MRR_DEPTH) / count)


def embedder_stats(result: EmbedderResult) -> dict[str, RetrievalStats]:
    """Hit@1, hit@5 and mean reciprocal rank in English, in Chinese, and over all questions."""
    by_language = {language: _stats([q for q in result.questions if q.language == language]) for language in LANGUAGES}
    return {**by_language, "all": _stats(result.questions)}


def write_embedder_report(results: list[EmbedderResult], out_dir: Path) -> tuple[Path, Path]:
    """evals/results/embedders-<UTC timestamp>.json with every question's rank, and a Markdown summary beside it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path, md_path = out_dir / f"embedders-{stamp}.json", out_dir / f"embedders-{stamp}.md"
    json_path.write_text(json.dumps([{
        "model": r.model, "dim": r.dim, "licence": r.licence, "error": r.error, "corpus_seconds": round(r.corpus_seconds, 1),
        "query_seconds": round(r.query_seconds, 1),
        "questions": [{"topic": q.topic_id, "language": q.language, "rank": q.rank, "top_url": q.top_url}
                      for q in r.questions]} for r in results], ensure_ascii=False, indent=1), encoding="utf-8")
    rows = []
    for r in results:
        if r.error:
            rows.append(f"| {r.model} | {r.dim} | {r.licence} | failed: {r.error} | | | | | |")
            continue
        s = embedder_stats(r)
        rows.append(f"| {r.model} | {r.dim} | {r.licence} | {s['all'].hit_at_1:.0%} | {s['all'].hit_at_5:.0%} | "
                    f"{s['all'].mrr:.2f} | {s['en'].hit_at_1:.0%} / {s['zh'].hit_at_1:.0%} | "
                    f"{s['en'].hit_at_5:.0%} / {s['zh'].hit_at_5:.0%} | {r.corpus_seconds + r.query_seconds:.0f} s |")
    ok = [r for r in results if not r.error]
    topics = list(dict.fromkeys(q.topic_id for r in ok for q in r.questions))
    ranks = {(r.model, q.topic_id, q.language): q.rank for r in ok for q in r.questions}
    grid = ["| Topic | " + " | ".join(f"{r.model.split('/')[-1]} EN / ZH" for r in ok) + " |",
            "|---|" + "---|" * len(ok),
            *(f"| {t} | " + " | ".join(f"{ranks[(r.model, t, 'en')] or '–'} / {ranks[(r.model, t, 'zh')] or '–'}"
                                       for r in ok) + " |" for t in topics)]
    md = ["# Embedding comparison", "",
          f"Started {results[0].started_at if results else datetime.now(UTC).isoformat(timespec='seconds')}.", "",
          ("Every model embedded the same documents, as lom setup stores them, and answered the same questions, each "
           "asked in English and in Chinese (evals/retrieval_cases.yaml). Documents are ranked by search_documents' "
           "vector rankings, chunks and document keys fused by reciprocal rank; its full-text ranking is the same for "
           f"every model and is left out. Hit@1 and hit@{HIT_DEPTH} count questions whose answer ranks first or in "
           f"the top {HIT_DEPTH}; MRR is the mean reciprocal rank of the first answer (0 below rank {MRR_DEPTH}). "
           "Time covers embedding the corpus and the questions on this machine's CPU."), "",
          (f"| Model | Dimensions | Licence | Hit@1 | Hit@{HIT_DEPTH} | MRR | Hit@1 EN / ZH | Hit@{HIT_DEPTH} EN / ZH | "
           "Time |"), "|---|---|---|---|---|---|---|---|---|", *rows, "", "## Rank of the first answer", "", *grid, ""]
    md_path.write_text("\n".join(md), encoding="utf-8")
    return json_path, md_path
