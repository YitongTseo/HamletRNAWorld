"""Pluggable embedding system.

Each `Embedder` turns a tokenized sentence (list[str]) into a (N, D) numpy
array of contextual word embeddings — one row per input word. Subword
tokenization inside any model is averaged back to the word level.

Switching embedders is a one-line change in main.py / World construction.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Anything with `embed_sentence(words) -> (N, D) array` and a `dim` property."""

    @property
    def dim(self) -> int: ...

    @property
    def name(self) -> str: ...

    def embed_sentence(self, words: list[str]) -> np.ndarray: ...

    def embed_sentences(self, sentences: list[list[str]]) -> list[np.ndarray]:
        """Default: embed sentence-by-sentence. Implementations may override
        with a batched path if they want."""
        return [self.embed_sentence(s) for s in sentences]


# Re-exports kept lazy: importing transformers/torch is slow, so the user only
# pays the cost when they actually request DistilBert. RandomEmbedder is cheap.
from .random_embedder import RandomEmbedder  # noqa: E402

__all__ = ["Embedder", "RandomEmbedder", "load"]


def load(name: str = "distilbert", **kwargs) -> Embedder:
    """Factory. Knows about the embedders shipped in this package."""
    if name == "distilbert":
        from .distilbert import DistilBertEmbedder
        return DistilBertEmbedder(**kwargs)
    if name == "random":
        return RandomEmbedder(**kwargs)
    raise ValueError(f"unknown embedder {name!r}")
