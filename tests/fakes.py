"""Test doubles that stand in for slow or external dependencies."""
import hashlib
import math
import re

WORD = re.compile(r"\w+")


class FakeEmbedder:
    """A deterministic bag-of-words embedder: each lowercase word hashes to one dimension, then the vector is
    L2-normalized. Texts that share words get a higher cosine, which is enough for de-duplication and ranking tests
    without downloading the real model. Queries and documents are embedded the same way."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in WORD.findall(text.lower()):
            vec[int(hashlib.sha1(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0], norm = 1.0, 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class ConstantEmbedder:
    """Every text, query or document, gets the same unit vector. The vector rankings then tie (lower id first), so a
    search test can isolate what keyword matching contributes."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _embed(self) -> list[float]:
        return [1.0] + [0.0] * (self.dim - 1)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed() for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed()
