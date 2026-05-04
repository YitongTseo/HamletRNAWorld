"""Deterministic random-vector embedder. Useful offline / for tests.

Each unique word string maps to a stable random unit vector (seeded by hash
of the word). Identical words across sentences get identical vectors — i.e.
this embedder is NOT contextual. It's a baseline for sanity-checking the
pipeline before downloading a real model.
"""
from __future__ import annotations

import hashlib

import numpy as np


class RandomEmbedder:
    name = "random"

    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_sentence(self, words: list[str]) -> np.ndarray:
        out = np.empty((len(words), self._dim), dtype=np.float32)
        for i, w in enumerate(words):
            out[i] = self._vec(w)
        return out

    def embed_sentences(self, sentences: list[list[str]]) -> list[np.ndarray]:
        return [self.embed_sentence(s) for s in sentences]

    def _vec(self, word: str) -> np.ndarray:
        h = hashlib.blake2b(word.lower().encode("utf-8"), digest_size=16).digest()
        seed = int.from_bytes(h, "little") & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dim).astype(np.float32)
        v /= np.linalg.norm(v) + 1e-9
        return v
