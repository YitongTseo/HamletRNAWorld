"""Strand: one tokenized sentence expanded to character-level beads.

Each Strand carries:
- the original sentence text and per-letter character / token-id arrays
- the list of tokens (words + punctuation) and their precomputed embeddings
- a per-token `reactive` flag (False for stop words / punctuation)

The simulator works at the LETTER level for physics (so word length matters)
but at the TOKEN level for affinity (so semantics drive bonding).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from corpus.hamlet import is_non_reactive
from corpus.letterize import Letterized, letterize


@dataclass
class Strand:
    text: str                       # original sentence
    chars: list[str]                # length L (per letter)
    token_ids: np.ndarray           # (L,) int — which token each letter belongs to, or -1
    tokens: list[str]               # length T
    embeddings: np.ndarray          # (T, D) — one per token, contextual
    reactive: np.ndarray            # (T,) bool — True if this token participates in pair bonds

    @property
    def n_letters(self) -> int:
        return len(self.chars)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)


def make_strands(sentences_raw: list[str], tokens_per_sentence: list[list[str]],
                 embedder) -> list[Strand]:
    """Build Strands from original sentence text + pretokenized tokens + embedder.

    sentences_raw:        the original sentence strings (used for letter alignment)
    tokens_per_sentence:  matching tokenization (one list per sentence)
    embedder:             any object with `embed_sentences(list[list[str]])`
    """
    if len(sentences_raw) != len(tokens_per_sentence):
        raise ValueError("sentences_raw and tokens_per_sentence must align")

    embs = embedder.embed_sentences(tokens_per_sentence)

    out: list[Strand] = []
    for raw, tokens, emb in zip(sentences_raw, tokens_per_sentence, embs):
        if emb.shape[0] != len(tokens):
            raise ValueError(
                f"embedder returned {emb.shape[0]} vectors for {len(tokens)} tokens"
            )
        L: Letterized = letterize(raw, tokens)
        reactive = np.array(
            [not is_non_reactive(t) for t in tokens], dtype=bool
        )
        out.append(Strand(
            text=raw,
            chars=list(L.chars),
            token_ids=np.array(L.token_ids, dtype=np.int32),
            tokens=list(tokens),
            embeddings=emb.astype(np.float32),
            reactive=reactive,
        ))
    return out


# ----------------------------------------------------------------------
# Affinity & color helpers — operate on TOKEN embeddings (not letter-level).
# ----------------------------------------------------------------------

def compute_token_affinity(token_embeddings: np.ndarray) -> np.ndarray:
    """Pairwise (1 - cos_sim)/(rescaled to [0,1]) over all tokens in the world."""
    if token_embeddings.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    e = token_embeddings.astype(np.float32)
    n = e.shape[0]
    e_norm = np.linalg.norm(e, axis=1, keepdims=True)
    e_unit = e / np.clip(e_norm, 1e-9, None)
    cos = e_unit @ e_unit.T
    np.fill_diagonal(cos, 1.0)
    off = ~np.eye(n, dtype=bool)
    if off.any():
        c_min = float(cos[off].min())
        c_max = float(cos[off].max())
    else:
        c_min, c_max = -1.0, 1.0
    span = c_max - c_min
    if span < 1e-6:
        return np.zeros_like(cos)
    aff = (c_max - cos) / span
    np.fill_diagonal(aff, 0.0)
    return aff.astype(np.float32)


def token_embeddings_to_colors(token_embeddings: np.ndarray) -> np.ndarray:
    """PCA-3 → bright, saturated RGB. Returns (T, 3) f32."""
    n, d = token_embeddings.shape
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    e = token_embeddings.astype(np.float32) - token_embeddings.mean(axis=0, keepdims=True)
    if n == 1 or d == 0:
        return np.full((n, 3), 0.7, dtype=np.float32)
    _, _, vt = np.linalg.svd(e, full_matrices=False)
    k = min(3, vt.shape[0])
    proj = e @ vt[:k].T
    if k < 3:
        proj = np.concatenate([proj, np.zeros((n, 3 - k), dtype=np.float32)], axis=1)
    p_min = proj.min(axis=0, keepdims=True)
    p_max = proj.max(axis=0, keepdims=True)
    norm = (proj - p_min) / np.clip(p_max - p_min, 1e-9, None)
    rgb = 0.30 + 0.70 * norm
    return rgb.astype(np.float32)
