import math

import pytest

from london_office_monitor.embed import get_embedder


@pytest.mark.embedding
def test_real_embedder_gives_normalized_vectors_and_instructs_queries(settings):
    """Characterizes the fastembed + Qwen3 assumptions from spike Q10: 1024 normalized dimensions, and the query
    instruction actually changes the query vector (fastembed adds none itself)."""
    embedder = get_embedder(settings)
    doc = embedder.embed_documents(["AI company leases offices in King's Cross"])[0]
    query = embedder.embed_query("AI company leases offices in King's Cross")
    assert len(doc) == settings.embedding_dim and math.isclose(math.sqrt(sum(v * v for v in doc)), 1.0, rel_tol=1e-5)
    assert doc != query
