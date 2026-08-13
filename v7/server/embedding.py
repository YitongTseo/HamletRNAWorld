"""v7 learned, context-aware word embedding + POS syntax net.

Replaces v6's frozen 12-dim UMAP with a small neural network that is
CO-EVOLVED with the worms by the same NES optimizer. See
`docs/superpowers/specs/2026-07-08-v7-learned-embedding-design.md`.

Two nets, whose parameters together form one flat "shared genome":

  Embedding (word -> 11-dim chemosensory drive, PC0..PC10)
    E : Linear(512 -> 11) + ReLU     applied to the current word AND each of
                                     the last-5 eaten words (SAME weights)
    Hh: Linear(55 -> 11)  + ReLU     summarizes the 5 eaten-word encodings
    Hf: Linear(22 -> 11)  -> sigmoid fuses E(current) with the history summary

  POS syntax (current+history POS -> 1-dim, PC11)
    P1: Linear(72 -> 16)  + ReLU     72 = 6 slots (current + 5 history) x 12 tags
    P2: Linear(16 -> 1)   -> sigmoid

The 512-dim word vectors are the FROZEN `nomic-embed-text-v1.5` embeddings,
built offline into `cache/corpus_nomic512.json` by
`scripts/build_corpus_nomic512.py`. Only the small nets above learn.

Memory: the worm's last-5 EATEN words feed the history inputs, so the same
on-screen word produces a different chemosensory vector depending on what the
worm recently ate. This REPLACES v6's decaying-residual afterglow.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from server import pos_grammar, pos_scorers

V7_ROOT = Path(__file__).resolve().parent.parent
_NOMIC_PATH = V7_ROOT / "cache" / "corpus_nomic512.json"
_UMAP_PATH = V7_ROOT / "cache" / "corpus_umap.json"

# --- encoder mode ----------------------------------------------------------
# "umap"    (DEFAULT) — PC0..PC10 are the frozen v6 UMAP coordinates.
# "learned" — PC0..PC10 come from the v7 co-evolved net below.
#
# 2026-08-13: the default reverted to UMAP after the learned encoder was
# measured on 101 live generations and found to have gone nearly blind. Two
# compounding causes, both in the numbers:
#
#   * He init (`sqrt(2/fan_in)`) assumes unit-variance inputs, but nomic-512
#     vectors are L2-NORMALISED, so per-component std is 0.044 — a 23x
#     deficit. Pre-activations land tiny and the sigmoid never spreads.
#   * Different nomic word vectors have mean pairwise cosine 0.700, and the
#     shared mean vector is 84% of a typical word's norm. Only ~16% of each
#     vector is word-specific. A random linear projection PRESERVES that
#     common component; UMAP exists precisely to strip it and re-spread.
#
# Measured across 4000 corpus words: the learned channels had per-channel
# std 0.027 over the range 0.40-0.55, against UMAP's std 0.131 over the full
# 0-1. Since `compute_pca_activation` feeds the channel value straight to
# neuron firing with no renormalisation, word identity was ~4% of the drive
# and the rest a DC offset — the worms could not smell one word from another.
# The (1+1)-ES that was meant to learn its way out of this accepted 2-7
# mutations in 101 generations (zero in the last 30 for six of eight flasks).
ENCODER_UMAP = "umap"
ENCODER_LEARNED = "learned"


def encoder_mode() -> str:
    m = os.environ.get("WORMLET_ENCODER", ENCODER_UMAP).strip().lower()
    return m if m in (ENCODER_UMAP, ENCODER_LEARNED) else ENCODER_UMAP


# --- dimensions (load-bearing; the genome layout depends on these) ---------
D_IN = 512          # frozen nomic embedding width
D_EMB = 11          # PC0..PC10 (the 12th, PC11, is the POS grammar channel)
HISTORY = 5         # number of eaten words used as context
D_HIST_CONCAT = D_EMB * HISTORY   # 55
D_FUSE = D_EMB * 2                 # 22  (E(current) ++ history summary)

# Canonical universal POS tags -> one-hot index. Unknown tags map to "X".
# Retained because the sanity-experiment scorers and the viewer still use this
# ordering; the PC11 channel itself no longer one-hots anything.
POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "DET", "ADP",
            "PRON", "PRT", "CONJ", "NUM", "X", "."]
N_TAGS = len(POS_TAGS)            # 12
_POS_INDEX = {t: i for i, t in enumerate(POS_TAGS)}


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


# Ordered layout of every weight/bias block in the flat genome. Each entry is
# (name, shape). flatten()/from_flat() walk this list in order, so the genome
# vector is fully determined by _LAYOUT — never reorder it once runs exist.
#
# 2026-08-13: the POS sub-net (W_P1/b_P1/W_P2/b_P2, 1185 floats) was REMOVED
# from the layout. PC11 now comes from `server.pos_grammar`, a deterministic
# corpus-derived POS trigram, so there is nothing left to learn there. Those
# blocks were the LAST entries in the layout, so a legacy 7697-float genome is
# a valid 6512-float genome followed by dead weight — `from_flat` therefore
# accepts the longer vector and ignores the tail, which keeps old
# `embedder.json` files loadable for comparison.
_LAYOUT: list[tuple[str, tuple[int, ...]]] = [
    ("W_E",  (D_IN, D_EMB)),   ("b_E",  (D_EMB,)),
    ("W_Hh", (D_HIST_CONCAT, D_EMB)), ("b_Hh", (D_EMB,)),
    ("W_Hf", (D_FUSE, D_EMB)), ("b_Hf", (D_EMB,)),
]

GENOME_SIZE = sum(int(np.prod(shape)) for _n, shape in _LAYOUT)  # 6,512
LEGACY_GENOME_SIZE = 7697  # v7.0/v7.1, with the POS sub-net appended


@dataclass
class EmbeddingParams:
    """The learnable weights of both nets, as numpy arrays."""
    W_E: np.ndarray
    b_E: np.ndarray
    W_Hh: np.ndarray
    b_Hh: np.ndarray
    W_Hf: np.ndarray
    b_Hf: np.ndarray

    def flatten(self) -> np.ndarray:
        return np.concatenate([getattr(self, n).ravel() for n, _s in _LAYOUT])

    @classmethod
    def from_flat(cls, vec: np.ndarray) -> "EmbeddingParams":
        vec = np.asarray(vec, dtype=np.float64)
        kw: dict[str, np.ndarray] = {}
        i = 0
        for name, shape in _LAYOUT:
            n = int(np.prod(shape))
            kw[name] = vec[i:i + n].reshape(shape)
            i += n
        # A legacy genome carries the removed POS sub-net as a tail; ignore it.
        if i != vec.shape[0] and vec.shape[0] != LEGACY_GENOME_SIZE:
            raise ValueError(f"genome size {vec.shape[0]} != expected {i}")
        return cls(**kw)

    @classmethod
    def random_init(cls, seed: int = 0) -> "EmbeddingParams":
        """He-ish random init. Small enough that a fresh net produces a mild,
        non-saturated signal; NES takes it from there."""
        rng = np.random.default_rng(seed)
        kw: dict[str, np.ndarray] = {}
        for name, shape in _LAYOUT:
            if name.startswith("b_"):
                kw[name] = np.zeros(shape, dtype=np.float64)
            else:
                fan_in = shape[0]
                scale = np.sqrt(2.0 / fan_in)
                kw[name] = rng.standard_normal(shape) * scale
        return cls(**kw)


def _onehot_pos(tag: str | None) -> np.ndarray:
    v = np.zeros(N_TAGS, dtype=np.float64)
    if tag is None:
        return v            # padding slot: no tag
    idx = _POS_INDEX.get(tag, _POS_INDEX["X"])
    v[idx] = 1.0
    return v


def _norm(word: str) -> str:
    """The canonical corpus/nomic lookup key: lowercased, apostrophes and
    surrounding whitespace stripped. MUST match how `_load_nomic` keys the
    table and how `_encode`/`vec512` look words up."""
    return word.lower().strip("'").strip() if word else ""


class EmbeddingModel:
    """Holds one set of params + the frozen nomic table, and runs the forward
    pass. Params are constant within a generation, so per-word encodings are
    cached and the cache is cleared whenever params change."""

    def __init__(self, params: EmbeddingParams, nomic: dict[str, np.ndarray] | None = None,
                 mode: str | None = None):
        self.mode = mode or encoder_mode()
        self.params = params
        # In UMAP mode the frozen 11-dim coordinates ARE the encoder, so the
        # 512-dim nomic table is never consulted and isn't loaded (it's a 39 MB
        # blob). `_nomic` stays populated only on the learned path.
        if self.mode == ENCODER_UMAP:
            self._nomic = {}
            self._umap = _load_umap()
        else:
            self._nomic = nomic if nomic is not None else _load_nomic()
            self._umap = {}
        self._grammar = pos_grammar.get_grammar()
        self._enc_cache: dict[str, np.ndarray] = {}
        # Precompute tables (shared across every worm in a flask; rebuilt when
        # params change). Populated lazily by prime().
        self._words: list[str] | None = None      # corpus keys (already _norm'd)
        self._widx: dict[str, int] = {}           # word -> row in _E_table
        self._vecs: np.ndarray | None = None      # (N, D_IN) frozen nomic matrix
        self._E_table: np.ndarray | None = None   # (N, D_EMB) = ReLU(vecs @ W_E + b_E)
        self.prime()

    # ---- params ----
    def set_params(self, params: EmbeddingParams) -> None:
        self.params = params
        self._enc_cache.clear()
        self.prime()

    def set_genome(self, vec: np.ndarray) -> None:
        """No-op in UMAP mode — the encoder is frozen and has no genome. The
        callers (flask_embedder / the rollover path) stay unchanged; they just
        stop having any effect."""
        if self.mode == ENCODER_UMAP:
            return
        self.set_params(EmbeddingParams.from_flat(vec))

    # ---- precompute (once per generation per flask) ----
    def prime(self) -> None:
        """Precompute the E: 512→11 projection over the WHOLE corpus under the
        current W_E/b_E. Because the embedder is shared across a flask and fixed
        within a generation, this runs once per generation per flask; every
        worm's per-tick smell-pass then reads the table instead of re-running
        the 512→11 matmul per word. The frozen nomic matrix + index are built
        once and reused across primings."""
        if self.mode == ENCODER_UMAP:
            # Frozen table: build the (N, 11) coordinate matrix once. Nothing
            # depends on params, so re-priming is idempotent and free.
            if self._words is None:
                self._words = list(self._umap.keys())
                self._widx = {w: i for i, w in enumerate(self._words)}
                self._E_table = (np.stack([self._umap[w] for w in self._words])
                                 if self._words else np.zeros((0, D_EMB)))
            return
        if not self._nomic:
            self._words = []
            self._widx = {}
            self._vecs = np.zeros((0, D_IN), dtype=np.float64)
            self._E_table = np.zeros((0, D_EMB), dtype=np.float64)
            return
        if self._words is None:
            self._words = list(self._nomic.keys())
            self._widx = {w: i for i, w in enumerate(self._words)}
            self._vecs = np.stack([self._nomic[w] for w in self._words])
        self._E_table = _relu(self._vecs @ self.params.W_E + self.params.b_E)

    # ---- lookups ----
    def vec512(self, word: str) -> np.ndarray | None:
        """Frozen 512-dim nomic vector for a word, or None if OOV."""
        if not word:
            return None
        return self._nomic.get(word.lower().strip("'").strip())

    def _encode(self, word: str) -> np.ndarray:
        """E(word) -> (11,) with ReLU. OOV -> zeros. Cached per generation."""
        key = word.lower().strip("'").strip() if word else ""
        cached = self._enc_cache.get(key)
        if cached is not None:
            return cached
        v = self._nomic.get(key)
        if v is None:
            out = np.zeros(D_EMB, dtype=np.float64)
        else:
            out = _relu(v @ self.params.W_E + self.params.b_E)
        self._enc_cache[key] = out
        return out

    # ---- forward pass ----
    def embed(self, current_word: str, history_words: list[str]) -> np.ndarray | None:
        """Return the 12-dim chemosensory drive for `current_word` given the
        worm's recent eaten context, or None if the current word is OOV.

        history_words: most-recent-first, length 0..HISTORY. Shorter histories
        (and OOV history words) are zero-padded.
        """
        # PC11 is the same in both modes: how grammatical this word's POS is as
        # a continuation of the POS sequence the worm has already eaten.
        pos_out = np.array([self._grammar.fit_word(current_word, list(history_words))])

        if self.mode == ENCODER_UMAP:
            row = self._umap.get(_norm(current_word))
            if row is None:
                return None
            return np.concatenate([row, pos_out])

        cur512 = self.vec512(current_word)
        if cur512 is None:
            return None
        p = self.params

        # --- embedding branch ---
        hist = (list(history_words) + [""] * HISTORY)[:HISTORY]
        hist_embs = np.concatenate([self._encode(w) for w in hist])   # (55,)
        hist_summary = _relu(hist_embs @ p.W_Hh + p.b_Hh)             # (11,)
        cur_emb = _relu(cur512 @ p.W_E + p.b_E)                        # (11,)
        fuse_in = np.concatenate([cur_emb, hist_summary])             # (22,)
        emb_out = _sigmoid(fuse_in @ p.W_Hf + p.b_Hf)                 # (11,)

        return np.concatenate([emb_out, pos_out])                     # (12,)

    # ---- vectorized forward pass (the hot path) ----
    def embed_batch(
        self, current_words: list[str], history_words: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Embed MANY current words that share ONE eaten-word history, in a
        couple of matmuls. This is the per-worm, per-brain-tick smell pass:
        every in-range on-screen word is a `current_word`; they all see the
        same worm history, so the history summary is computed exactly once.

        Returns (out, valid):
          out   : (K, 12) float64 — the chemosensory drive per current word.
          valid : (K,) bool       — False where the word is OOV (out rows are
                                     zeroed). Callers skip invalid rows, exactly
                                     as `embed()` returns None for an OOV word.

        Numerically identical to calling `embed()` per word (verified to 1e-9).
        """
        K = len(current_words)
        if K == 0:
            return np.zeros((0, D_EMB + 1), dtype=np.float64), np.zeros(0, dtype=bool)
        p = self.params
        E = self._E_table
        if E is None:
            self.prime()
            E = self._E_table

        ci = np.fromiter((self._widx.get(_norm(w), -1) for w in current_words),
                         dtype=np.int64, count=K)
        valid = ci >= 0
        safe = np.where(valid, ci, 0)

        # --- PC11: POS-grammar fit (both modes) ---
        # The whole batch shares one history, so the normalised tag->fit map is
        # looked up ONCE and then indexed per candidate word: K dict lookups,
        # no matmul. This is why the channel is free relative to the old net.
        fit_by_tag = self._grammar.context_distribution(list(history_words))
        pos_out = np.fromiter(
            (fit_by_tag.get(self._grammar.tag(w), 0.0) for w in current_words),
            dtype=np.float64, count=K,
        ).reshape(K, 1)

        if self.mode == ENCODER_UMAP:
            emb_out = np.where(valid[:, None], E[safe], 0.0)                      # (K,11)
            out = np.concatenate([emb_out, pos_out], axis=1)                      # (K,12)
            out[~valid] = 0.0
            return out, valid

        # --- history (computed ONCE for all K words) ---
        hist = (list(history_words) + [None] * HISTORY)[:HISTORY]
        # History E-encodings: table row by _norm, zeros on OOV/pad — matches
        # `_encode()` (which returns zeros for OOV / empty).
        hidx = [self._widx.get(_norm(w), -1) if w else -1 for w in hist]
        he = np.concatenate([E[h] if h >= 0 else np.zeros(D_EMB) for h in hidx])  # (55,)
        hist_summary = _relu(he @ p.W_Hh + p.b_Hh)                                # (11,)

        # --- embedding branch (batched) ---
        cur_emb = np.where(valid[:, None], E[safe], 0.0)                          # (K,11)
        fuse = np.concatenate(
            [cur_emb, np.broadcast_to(hist_summary, (K, D_EMB))], axis=1)         # (K,22)
        emb_out = _sigmoid(fuse @ p.W_Hf + p.b_Hf)                                # (K,11)

        out = np.concatenate([emb_out, pos_out], axis=1)                          # (K,12)
        out[~valid] = 0.0
        return out, valid


# --- frozen UMAP table ------------------------------------------------------

_UMAP_CACHE: dict[str, np.ndarray] | None = None


def _load_umap() -> dict[str, np.ndarray]:
    """{lowercased_word: np.array(11)} from cache/corpus_umap.json.

    The artifact holds 12 UMAP dimensions; we take the FIRST 11 as PC0..PC10
    and hand PC11 to the POS grammar. Coordinates are already min-max scaled
    into [0,1] per dimension by scripts/build_corpus_umap.py, which is exactly
    the range `compute_pca_activation` wants — it multiplies the channel value
    straight into neuron firing with no renormalisation.

    Missing file -> empty map (chemosensation goes silent; smoke tests run
    without it), matching _load_nomic's behaviour."""
    global _UMAP_CACHE
    if _UMAP_CACHE is not None:
        return _UMAP_CACHE
    if not _UMAP_PATH.exists():
        _UMAP_CACHE = {}
        return _UMAP_CACHE
    data = json.loads(_UMAP_PATH.read_text())
    words = data["words"]
    coords = data["umap12"]
    _UMAP_CACHE = {
        _norm(w): np.asarray(c[:D_EMB], dtype=np.float64)
        for w, c in zip(words, coords)
    }
    return _UMAP_CACHE


# --- frozen nomic table -----------------------------------------------------

_NOMIC_CACHE: dict[str, np.ndarray] | None = None


def _load_nomic() -> dict[str, np.ndarray]:
    """{lowercased_word: np.array(512)}. Loaded once per process. Missing file
    -> empty map (chemosensation goes silent; smoke tests run without it)."""
    global _NOMIC_CACHE
    if _NOMIC_CACHE is not None:
        return _NOMIC_CACHE
    if not _NOMIC_PATH.exists():
        _NOMIC_CACHE = {}
        return _NOMIC_CACHE
    data = json.loads(_NOMIC_PATH.read_text())
    words = data["words"]
    vecs = data.get("nomic512") or data["vectors"]
    _NOMIC_CACHE = {w: np.asarray(v, dtype=np.float64) for w, v in zip(words, vecs)}
    return _NOMIC_CACHE


# --- process-wide model singleton (shared across all worms in this process) -

_MODEL: EmbeddingModel | None = None


def get_model() -> EmbeddingModel:
    """The process-wide shared embedding model. Cold-starts from a random init
    seeded by WORMLET_EMBEDDING_SEED (default 0) so a fresh process is
    deterministic. app.py replaces its genome from the shared-net coordinator
    each generation via set_model_genome()."""
    global _MODEL
    if _MODEL is None:
        seed = int(os.environ.get("WORMLET_EMBEDDING_SEED", "0"))
        _MODEL = EmbeddingModel(EmbeddingParams.random_init(seed))
    return _MODEL


def set_model_genome(vec: np.ndarray) -> None:
    get_model().set_genome(vec)
