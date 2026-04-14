"""
Word-level semantic embeddings using sentence-transformers.

Each unique non-filler word is embedded independently.
Cosine similarities are pre-computed and stored in a lookup dict
so the physics engine can query them cheaply.

Filler words always return cosine_sim = 0 (neutral).
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple

from config import EMBED_MODEL, EMBED_CACHE, COS_NEUTRAL


EmbeddingMap = Dict[str, np.ndarray]          # word → unit vector
SimCache     = Dict[Tuple[str, str], float]   # (w1, w2) → cosine similarity


def _load_model():
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model '{EMBED_MODEL}'…")
    model = SentenceTransformer(EMBED_MODEL)
    return model


def compute_embeddings(words: List[str]) -> EmbeddingMap:
    """
    Compute L2-normalised embeddings for each word.
    Uses on-disk cache to avoid recomputation.
    """
    os.makedirs("cache", exist_ok=True)

    if os.path.exists(EMBED_CACHE):
        with open(EMBED_CACHE, "rb") as f:
            cached: EmbeddingMap = pickle.load(f)
        # Check if all words are already cached
        missing = [w for w in words if w not in cached]
        if not missing:
            print(f"All {len(words)} embeddings loaded from cache.")
            return {w: cached[w] for w in words}
        print(f"Cache hit for {len(words)-len(missing)}/{len(words)} words; "
              f"computing {len(missing)} new embeddings…")
    else:
        cached = {}
        missing = words

    model = _load_model()
    batch_size = 256
    new_vecs = {}
    for i in range(0, len(missing), batch_size):
        batch = missing[i:i + batch_size]
        vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        # L2 normalise
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        vecs = vecs / norms
        for w, v in zip(batch, vecs):
            new_vecs[w] = v

    cached.update(new_vecs)
    with open(EMBED_CACHE, "wb") as f:
        pickle.dump(cached, f)
    print(f"Saved {len(cached)} embeddings to cache.")

    return {w: cached[w] for w in words}


def build_sim_cache(emb_map: EmbeddingMap) -> SimCache:
    """
    Pre-compute pairwise cosine similarities for all unique semantic words.
    Stored as (w1, w2) with w1 <= w2 lexicographically.
    """
    sim_cache_path = "cache/sim_cache.pkl"
    if os.path.exists(sim_cache_path):
        with open(sim_cache_path, "rb") as f:
            data = pickle.load(f)
        if data.get("words_key") == frozenset(emb_map.keys()):
            print(f"Pairwise sim cache loaded ({len(data['sims'])} pairs).")
            return data["sims"]

    words = sorted(emb_map.keys())
    n = len(words)
    print(f"Computing {n*(n-1)//2} pairwise cosine similarities…")
    mat = np.stack([emb_map[w] for w in words])  # (n, d)
    # cosine similarity matrix (embeddings already unit-normalised)
    sim_mat = mat @ mat.T                          # (n, n)

    # ── Centre the distribution ──────────────────────────────────────────
    # all-MiniLM and similar models produce similarities in [0.1, 0.8] for
    # English word pairs; there are almost no negative raw cosines.
    # We subtract the global off-diagonal mean so that "more similar than
    # average" words repel and "less similar than average" words attract.
    # This creates meaningful secondary structure across Hamlet's vocabulary.
    mask = ~np.eye(n, dtype=bool)
    global_mean = float(sim_mat[mask].mean())
    sim_mat_centered = sim_mat - global_mean
    print(f"  Raw sim range: [{sim_mat[mask].min():.3f}, {sim_mat[mask].max():.3f}]"
          f"  mean={global_mean:.3f}")
    print(f"  Centered range: [{sim_mat_centered[mask].min():.3f},"
          f" {sim_mat_centered[mask].max():.3f}]")

    sims: SimCache = {}
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_mat_centered[i, j])
            # Zero out near-neutral zone
            if abs(s) < COS_NEUTRAL:
                s = 0.0
            if s != 0.0:
                sims[(words[i], words[j])] = s

    with open(sim_cache_path, "wb") as f:
        pickle.dump({"words_key": frozenset(words), "sims": sims}, f)
    print(f"Saved {len(sims)} non-neutral pairs to sim cache.")
    return sims


def get_cosine_sim(w1: str, w2: str, sim_cache: SimCache) -> float:
    """Look up cosine similarity; returns 0 for filler/unknown/neutral pairs."""
    key = (min(w1, w2), max(w1, w2))
    return sim_cache.get(key, 0.0)


class EmbeddingStore:
    """Convenience wrapper used by the simulation."""

    def __init__(self, words: List[str]):
        self.emb_map = compute_embeddings(words)
        self.sim_cache = build_sim_cache(self.emb_map)
        # Build integer-indexed arrays for fast Taichi lookup
        self._build_arrays(words)

    def _build_arrays(self, words: List[str]):
        """
        Create word→int index and a flat similarity table
        so the simulation can index by integer IDs rather than strings.
        """
        self.word_to_idx: Dict[str, int] = {w: i for i, w in enumerate(words)}
        n = len(words)
        # sim_table[i, j] = cosine similarity between words[i] and words[j]
        # filler words will never appear in this list
        self.sim_table = np.zeros((n, n), dtype=np.float32)
        for (w1, w2), s in self.sim_cache.items():
            i = self.word_to_idx.get(w1, -1)
            j = self.word_to_idx.get(w2, -1)
            if i >= 0 and j >= 0:
                self.sim_table[i, j] = s
                self.sim_table[j, i] = s
        self.n_semantic = n

    def sim_by_idx(self, i: int, j: int) -> float:
        if i < 0 or j < 0:
            return 0.0
        return float(self.sim_table[i, j])

    def compute_pca_colors(self) -> Dict[str, tuple]:
        """
        Map each semantic word to an RGB colour (0-255 ints) via 3-component PCA.
        Semantically similar words will have similar hues.
        Returns dict: word → (r, g, b).
        """
        from sklearn.decomposition import PCA

        words  = sorted(self.word_to_idx.keys())
        mat    = np.stack([self.emb_map[w] for w in words])  # (N, 384)

        pca    = PCA(n_components=3)
        proj   = pca.fit_transform(mat)          # (N, 3)

        # Normalize each PC to [0, 1]
        lo, hi = proj.min(axis=0), proj.max(axis=0)
        span   = np.where(hi - lo < 1e-9, 1.0, hi - lo)
        proj   = (proj - lo) / span              # (N, 3) in [0,1]

        # Increase saturation: push away from mid-grey
        proj   = np.clip(proj * 1.6 - 0.3, 0.0, 1.0)

        colors: Dict[str, tuple] = {}
        for w, rgb in zip(words, proj):
            r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            colors[w] = (r, g, b)
        return colors


if __name__ == "__main__":
    sample = ["death", "life", "love", "hate", "king", "ghost",
              "sword", "flower", "night", "day"]
    store = EmbeddingStore(sample)
    print("\nSample similarities:")
    pairs = [("death", "life"), ("love", "hate"), ("king", "ghost"),
             ("night", "day"), ("death", "flower")]
    for w1, w2 in pairs:
        s = store.sim_by_idx(store.word_to_idx[w1], store.word_to_idx[w2])
        print(f"  {w1:12s} ↔ {w2:12s}  cos={s:+.3f}")
