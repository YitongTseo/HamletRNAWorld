"""Build the per-word UMAP artifact — parallel to build_corpus_pca.py but
uses UMAP for dimensionality reduction instead of linear PCA.

Pipeline mirrors the PCA script:
1. Read full Hamlet, strip Gutenberg envelope, dedupe word types.
2. Embed each type with nomic-embed-text-v1.5 (same model as PCA build, so
   any cached weights are reused).
3. Fit UMAP twice:
   - n_components=12 with cosine metric → drives chemosensation (when we
     switch the sim over; not wired in yet).
   - n_components=2 → drives the focus-page hover scatter; this is the
     immediate user-visible change.
4. Min-max each output dim to [0,1] so the chemosensory neurons (excitatory
   only) can drive on them. Same softmax(β=5) sparsification as PCA build.
5. Save to cache/corpus_umap.json.

UMAP is stochastic; we pin random_state for reproducibility. Use cosine
metric because nomic returns L2-normalized vectors (so cosine and euclidean
are monotonically related, but cosine is the documented choice for text
embeddings).

Run:
    /home/web/.venv/bin/python scripts/build_corpus_umap.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

V6_ROOT = Path(__file__).resolve().parent.parent
SRC = V6_ROOT / "corpus" / "hamlet_gutenberg.txt"
OUT = V6_ROOT / "cache" / "corpus_umap.json"

N_DIMS = 12

# Same softmax sparsification as the PCA build. After min-max each dim is
# in [0,1], so every word has non-zero values across all 12 dims; summing
# multiple in-range words would saturate the chemosensory bars. Softmax
# concentrates each word on its signature dims and makes the per-word
# distribution sum to 1. β=5 matches the PCA build for comparability.
SOFTMAX_BETA = 5.0

# UMAP hyperparameters. Tuned for cluster visibility ("clumpiness"):
#   n_neighbors: lower → each point only sees its closest neighbors, so
#     local clumps stay tight at the cost of some global geometry. Default
#     is 15; 8 is a moderate clump-favoring value, 5 is aggressive.
#   min_dist: lower → output points within a cluster pack tighter. Default
#     is 0.1; 0.0 is the maximum-compression setting and is the standard
#     choice when "clusters visible" matters more than "smooth manifold".
#   metric=cosine: required for nomic's L2-normalized embeddings.
#   random_state: pins the stochastic init for reproducibility.
UMAP_N_NEIGHBORS = 8
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

START_RE = re.compile(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
END_RE = re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)


def load_play_text() -> str:
    raw = SRC.read_text(encoding="utf-8")
    start = START_RE.search(raw)
    end = END_RE.search(raw)
    if not start or not end:
        return raw
    return raw[start.end():end.start()]


def extract_unique_words(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z']+", text)
    seen: set[str] = set()
    ordered_unique: list[str] = []
    for t in tokens:
        lt = t.lower().strip("'")
        if not lt:
            continue
        if lt not in seen:
            seen.add(lt)
            ordered_unique.append(lt)
    return ordered_unique


def embed_words(words: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print("Loading nomic-embed-text-v1.5 (cached from PCA build)...")
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        device="cpu",
    )
    inputs = [f"clustering: {w}" for w in words]
    print(f"Embedding {len(inputs)} unique tokens on CPU...")
    t0 = time.monotonic()
    vecs = model.encode(
        inputs,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"Embedded in {time.monotonic() - t0:.1f}s. Shape: {vecs.shape}")
    return vecs


def umap_with_shift(vecs: np.ndarray, n_components: int) -> np.ndarray:
    import umap

    print(f"Fitting UMAP with n_components={n_components}, n_neighbors={UMAP_N_NEIGHBORS}, "
          f"min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC}...")
    t0 = time.monotonic()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )
    coords = reducer.fit_transform(vecs)
    print(f"  UMAP-{n_components} fit in {time.monotonic() - t0:.1f}s")
    # Min-max per dimension to [0, 1] — matches PCA build's normalization.
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    coords01 = (coords - lo) / (hi - lo + 1e-12)
    return coords01.astype(np.float64)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not SRC.exists():
        sys.exit(f"missing {SRC}")

    play = load_play_text()
    words = extract_unique_words(play)
    print(f"unique word types: {len(words)}")

    vecs = embed_words(words)

    coords12 = umap_with_shift(vecs, N_DIMS)
    coords2 = umap_with_shift(vecs, 2)

    # Softmax-sparsify the 12D version, same as PCA.
    ex = np.exp(SOFTMAX_BETA * coords12)
    coords12_sparse = ex / ex.sum(axis=1, keepdims=True)
    peaks = coords12_sparse.max(axis=1)
    meds = np.median(coords12_sparse, axis=1)
    ratios = peaks / np.maximum(meds, 1e-9)
    print(f"Softmax sparsified (β={SOFTMAX_BETA}); peak/median: "
          f"median={np.median(ratios):.2f}x  p10={np.percentile(ratios,10):.2f}x  "
          f"p90={np.percentile(ratios,90):.2f}x")

    # Reuse the NRC emotion lookup so the focus page's radar chart keeps
    # working whether it reads from the PCA or UMAP artifact.
    print("Looking up NRC emotion vectors per word...")
    sys.path.insert(0, str(V6_ROOT))
    from corpus.nrc_emotions import get_emotion_vector
    EMOTION_KEYS = ["joy", "trust", "anticipation", "surprise",
                    "fear", "disgust", "sadness", "anger"]
    emotion_rows = []
    n_with_emotion = 0
    for w in words:
        ev = get_emotion_vector(w)
        row = [round(float(ev.get(k, 0.0)), 3) for k in EMOTION_KEYS]
        if any(v > 0 for v in row):
            n_with_emotion += 1
        emotion_rows.append(row)
    print(f"  {n_with_emotion}/{len(words)} words have non-zero NRC emotion content")

    payload = {
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "prefix": "clustering:",
        "method": "umap",
        "n_words": len(words),
        "n_dims": N_DIMS,
        "umap_params": {
            "n_neighbors": UMAP_N_NEIGHBORS,
            "min_dist": UMAP_MIN_DIST,
            "metric": UMAP_METRIC,
            "random_state": UMAP_RANDOM_STATE,
        },
        "words": words,
        "umap12": [[round(float(x), 4) for x in row] for row in coords12],
        "umap12_sparse": [[round(float(x), 4) for x in row] for row in coords12_sparse],
        "softmax_beta": SOFTMAX_BETA,
        "umap2": [[round(float(x), 4) for x in row] for row in coords2],
        "emotion_keys": EMOTION_KEYS,
        "emotions": emotion_rows,
    }
    OUT.write_text(json.dumps(payload))
    print(f"Saved {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
