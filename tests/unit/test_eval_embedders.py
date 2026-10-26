import functools
import re

import numpy as np
import pytest
from typer.testing import CliRunner

from london_office_monitor.chunking import contextual_header, document_key_text
from london_office_monitor.cli import app
from london_office_monitor.evals import embedders as embedders_module
from london_office_monitor.evals import runner as runner_module
from london_office_monitor.evals.embedders import (EMBEDDER_SPECS, RETRIEVAL_CASES_FILE, build_corpus,
                                                   compare_embedders, embedder_stats, load_retrieval_cases,
                                                   rank_documents, write_embedder_report)
from london_office_monitor.ingest.store import content_hash
from london_office_monitor.paths import FIXTURES_DIR, INTERNAL_DIR
from london_office_monitor.seed.load import read_fixture_documents, read_internal

CJK = re.compile(r"[㐀-鿿]")
FIXTURE_DOCUMENTS = 193  # 186 news items, 1 Gazette notice and 6 synthetic items in data/fixtures
STORED_DOCUMENTS = 197   # what lom setup stores: the snapshot less its 2 exact duplicates, and 6 internal documents


@functools.cache
def corpus_docs():
    return read_fixture_documents(FIXTURES_DIR) + read_internal(INTERNAL_DIR)


def test_the_fixture_reader_returns_every_snapshot_document():
    docs = read_fixture_documents(FIXTURES_DIR)
    assert len(docs) == FIXTURE_DOCUMENTS and sum(d.synthetic for d in docs) == 6


def test_the_corpus_is_embedded_as_setup_embeds_it(settings):
    docs = corpus_docs()
    corpus = build_corpus(docs, settings)
    unique = []
    for doc in docs:  # store_documents skips an exact duplicate (same URL and title)
        if content_hash(doc) not in {content_hash(kept) for kept in unique}:
            unique.append(doc)
    assert len(unique) == STORED_DOCUMENTS and corpus.urls == [d.url for d in unique]
    assert corpus.key_texts == [document_key_text(d) for d in unique]
    first = docs[0]
    assert corpus.chunk_texts[0].startswith(contextual_header(first.publisher, first.title, first.subtitle, (),
                                                              first.published_at) + "\n")
    internal = [i for i, d in enumerate(docs) if d.source_kind == "internal"]
    assert sum(corpus.chunk_document.count(i) for i in internal) > len(internal)  # internal notes split by heading


def test_every_topic_is_asked_in_both_languages_and_answered_by_a_stored_document(settings):
    urls = set(build_corpus(corpus_docs(), settings).urls)
    topics = load_retrieval_cases(RETRIEVAL_CASES_FILE).topics
    assert len(topics) >= 15 and len({t.id for t in topics}) == len(topics)
    assert all(CJK.search(t.zh) and not CJK.search(t.en) for t in topics)
    assert all(url in urls for t in topics for url in t.relevant)


def test_ranking_fuses_the_chunk_and_key_rankings_as_search_does():
    # Chunk ranks A, B, C and key ranks B, C, A: by reciprocal rank (k=60) B scores 1/62+1/61, A 1/61+1/63, C 1/63+1/62.
    query = np.array([1.0, 0.0])
    chunks = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    keys = np.array([[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]])
    assert rank_documents(query, keys, chunks, [0, 1, 2]) == [1, 0, 2]


def test_a_document_is_ranked_by_its_best_chunk():
    query = np.array([1.0, 0.0])
    chunks = np.array([[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]])   # document 0 has a poor and an exact chunk
    keys = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert rank_documents(query, keys, chunks, [0, 0, 1]) == [0, 1]


def oracle_embed(corpus, wrong_in_chinese: bool):
    """A fake model: each document is its own axis, and a question points at its first answer, or in Chinese at the
    wrong document when `wrong_in_chinese`."""
    topics = load_retrieval_cases(RETRIEVAL_CASES_FILE).topics
    owner = {text: i for i, text in enumerate(corpus.key_texts)}
    owner.update({text: corpus.chunk_document[j] for j, text in enumerate(corpus.chunk_texts)})

    def embed(spec, texts):
        rows = []
        for text in texts:
            vector = np.zeros(len(corpus.urls))
            topic = next((t for t in topics if text.endswith((t.en, t.zh))), None)
            if topic is None:
                vector[owner[text.removeprefix(spec.document_prefix)]] = 1.0
            else:
                target = corpus.urls.index(topic.relevant[0])
                chinese = text.endswith(topic.zh)
                vector[(target + 1) % len(corpus.urls) if wrong_in_chinese and chinese else target] = 1.0
            rows.append(vector)
        return np.array(rows)
    return embed


def test_hits_and_reciprocal_rank_are_counted_per_language(settings):
    docs = corpus_docs()
    specs = [EMBEDDER_SPECS["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]]
    corpus = build_corpus(docs, settings)
    results = compare_embedders(specs, load_retrieval_cases(RETRIEVAL_CASES_FILE), corpus,
                                embed=oracle_embed(corpus, wrong_in_chinese=True))
    stats = embedder_stats(results[0])
    assert (stats["en"].hit_at_1, stats["en"].hit_at_5, stats["en"].mrr) == (1.0, 1.0, 1.0)
    assert stats["zh"].hit_at_1 == 0.0 and stats["all"].hit_at_1 == 0.5


def test_a_model_that_fails_is_reported_and_the_others_still_run(settings):
    docs = corpus_docs()
    corpus = build_corpus(docs, settings)
    good = oracle_embed(corpus, wrong_in_chinese=False)

    def embed(spec, texts):
        if spec.name.startswith("google/"):
            raise RuntimeError("model download failed")
        return good(spec, texts)
    specs = [EMBEDDER_SPECS["google/embeddinggemma-300m"], EMBEDDER_SPECS["Qwen/Qwen3-Embedding-0.6B-Q"]]
    results = compare_embedders(specs, load_retrieval_cases(RETRIEVAL_CASES_FILE), corpus, embed=embed)
    assert "download failed" in results[0].error and results[1].error is None
    assert embedder_stats(results[1])["all"].hit_at_1 == 1.0


def test_each_model_gets_its_own_query_and_document_prefixes(settings):
    seen = {}

    def embed(spec, texts):
        seen.setdefault(spec.name, []).extend(texts)
        return np.ones((len(texts), 2))
    docs = corpus_docs()[:3]
    corpus = build_corpus(docs, settings)
    cases = load_retrieval_cases(RETRIEVAL_CASES_FILE)
    compare_embedders([EMBEDDER_SPECS["google/embeddinggemma-300m"]], cases, corpus, embed=embed)
    texts = seen["google/embeddinggemma-300m"]
    assert f"title: none | text: {corpus.key_texts[0]}" in texts
    assert f"task: search result | query: {cases.topics[0].en}" in texts


def test_the_report_shows_each_model_and_each_question(settings, tmp_path):
    docs = corpus_docs()
    specs = [EMBEDDER_SPECS["Qwen/Qwen3-Embedding-0.6B-Q"]]
    cases = load_retrieval_cases(RETRIEVAL_CASES_FILE)
    corpus = build_corpus(docs, settings)
    results = compare_embedders(specs, cases, corpus, embed=oracle_embed(corpus, False))
    json_path, md_path = write_embedder_report(results, tmp_path)
    md = md_path.read_text(encoding="utf-8")
    assert json_path.name.startswith("embedders-") and "| Qwen/Qwen3-Embedding-0.6B-Q | 1024 |" in md
    assert all(t.id in md for t in cases.topics)


def test_eval_embedders_refuses_an_unknown_model():
    result = CliRunner().invoke(app, ["eval-embedders", "--models", "BAAI/unknown-model"])
    assert result.exit_code == 1 and "Unknown embedding model" in result.output


def test_eval_embedders_writes_its_report(monkeypatch, tmp_path, settings):
    corpus = build_corpus(corpus_docs(), settings)
    monkeypatch.setattr(embedders_module, "fastembed_texts", oracle_embed(corpus, wrong_in_chinese=False))
    monkeypatch.setattr(runner_module, "RESULTS_DIR", tmp_path)
    result = CliRunner().invoke(app, ["eval-embedders", "--models", "Qwen/Qwen3-Embedding-0.6B-Q"])
    assert result.exit_code == 0, result.output
    assert "Embedding comparison" in result.output and list(tmp_path.glob("embedders-*.md"))


@pytest.mark.parametrize("name", sorted(EMBEDDER_SPECS))
def test_every_candidate_names_its_dimensions(name):
    assert EMBEDDER_SPECS[name].dim in (384, 768, 1024)
