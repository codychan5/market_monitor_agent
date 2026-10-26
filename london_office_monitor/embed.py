"""Local embeddings (Qwen3-Embedding-0.6B through fastembed): no second API key, and Chinese questions retrieve
English text."""
import functools
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from .config import Settings
from .paths import from_project

# Qwen3-Embedding expects an instruction on queries but not on documents; fastembed adds none itself (spike Q10).
QUERY_INSTRUCTION = ("Instruct: Given a question about the London office market, retrieve relevant news passages\n"
                     "Query:")
EMBED_BATCH_SIZE = 32


class EmbeddingError(Exception):
    """The embedding model could not be loaded or returned vectors of the wrong size."""


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _normalized(vectors: Iterable, dim: int) -> list[list[float]]:
    out = []
    for vector in vectors:
        values = [float(v) for v in vector]
        if len(values) != dim:
            raise EmbeddingError(f"the embedding model returned {len(values)} dimensions, expected {dim}")
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        out.append([v / norm for v in values])
    return out


class FastEmbedEmbedder:
    """Loads the model on first use, so commands that never embed never pay for it."""

    def __init__(self, model_name: str, dim: int, cache_dir: Path):
        self.model_name = model_name
        self.dim = dim
        self.cache_dir = cache_dir
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding
                self._model = TextEmbedding(self.model_name, cache_dir=str(self.cache_dir))
            except Exception as e:  # noqa: BLE001  any download or ONNX failure becomes one actionable error
                raise EmbeddingError(f"could not load the embedding model {self.model_name} ({type(e).__name__}: "
                                     f"{e}); the first run needs internet to download about 1.1 GB") from e
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return _normalized(self._load().embed(texts, batch_size=EMBED_BATCH_SIZE), self.dim)

    def embed_query(self, text: str) -> list[float]:
        return _normalized(self._load().embed([QUERY_INSTRUCTION + text]), self.dim)[0]


@functools.cache
def _shared_embedder(model_name: str, dim: int, cache_dir: str) -> FastEmbedEmbedder:
    return FastEmbedEmbedder(model_name, dim, Path(cache_dir))


def get_embedder(settings: Settings) -> Embedder:
    """One embedder per process and model: loading it takes seconds, so every caller shares it."""
    return _shared_embedder(settings.embedding_model, settings.embedding_dim,
                            str(from_project(settings.embedding_cache_dir)))
