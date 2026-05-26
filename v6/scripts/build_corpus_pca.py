"""Build the per-word PCA artifact that drives v6 chemosensation.

Pipeline:
1. Read the full Hamlet text (corpus/hamlet_gutenberg.txt, Project Gutenberg
   public-domain edition).
2. Strip Gutenberg header/footer, keep only the play text.
3. Tokenize into word types (lowercased, deduplicated).
4. Embed each type with nomic-ai/nomic-embed-text-v1.5 using the documented
   `clustering:` task prefix.
5. PCA the (N, 768) matrix to 12 dimensions, min-max shift each dim to [0,1].
6. Also compute a 2D PCA for the focus-page hover scatter.
7. Save to data/corpus_pca.json — one-time preprocessing, never re-run at
   server boot.

This is a deterministic, idempotent build step. Run with:
    /home/web/.venv/bin/python scripts/build_corpus_pca.py
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
OUT = V6_ROOT / "cache" / "corpus_pca.json"

N_PCS = 12

# Gutenberg envelopes its texts with header/footer banners. Extract only
# what's between the START/END markers — anything else (license, blurb)
# would skew the embeddings toward modern English.
START_RE = re.compile(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
END_RE = re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)


def load_play_text() -> str:
    raw = SRC.read_text(encoding="utf-8")
    start = START_RE.search(raw)
    end = END_RE.search(raw)
    if not start or not end:
        # Fall back to using everything; better than crashing.
        return raw
    return raw[start.end():end.start()]


def extract_unique_words(text: str) -> list[str]:
    # Match word-like tokens. Keep apostrophes (e.g. "'tis"). Lowercase for
    # deduplication: 'Live' and 'live' get the same vector.
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
    """Returns (N, 768) numpy array; row i = embedding of words[i]."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading nomic-embed-text-v1.5 (first run downloads ~550MB)...")
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        device="cpu",
    )
    # Nomic's documented prefix: `clustering:` for dimensionality-reduction
    # / clustering-style tasks. PCA-driven chemosensation is exactly that.
    inputs = [f"clustering: {w}" for w in words]
    print(f"Embedding {len(inputs)} unique tokens on CPU (this will take a minute)...")
    t0 = time.monotonic()
    vecs = model.encode(
        inputs,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit-norm; PCA still makes sense
    )
    print(f"Embedded in {time.monotonic() - t0:.1f}s. Shape: {vecs.shape}")
    return vecs


def pca_with_shift(vecs: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center → SVD → project → min-max shift each PC to [0,1].
    Returns (coords, principal_axes, explained_variance_ratio)."""
    center = vecs.mean(axis=0)
    centered = vecs - center
    # full_matrices=False is the economy SVD — much cheaper for (5k, 768).
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    axes = Vt[:n_components]  # (n_components, 768)
    coords = centered @ axes.T  # (N, n_components)
    # Min-max per dimension to [0, 1]. We need this for the chemosensory
    # neurons (which only excite — they can't fire negatively).
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    coords01 = (coords - lo) / (hi - lo + 1e-12)
    var = (S ** 2) / max(1, (vecs.shape[0] - 1))
    explained = var[:n_components] / var.sum()
    return coords01, axes, explained


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not SRC.exists():
        sys.exit(f"missing {SRC} — `curl https://www.gutenberg.org/cache/epub/1524/pg1524.txt > {SRC}`")

    play = load_play_text()
    words = extract_unique_words(play)
    print(f"unique word types: {len(words)}")

    vecs = embed_words(words)

    print(f"PCA to {N_PCS} dimensions...")
    coords12, _, explained12 = pca_with_shift(vecs, N_PCS)
    print(f"  cumulative variance explained by {N_PCS} PCs: {explained12.sum():.3f}")
    print(f"  per-PC variance: {[round(x, 3) for x in explained12]}")

    # Also compute a 2D PCA for the visualizer hover scatter.
    coords2, _, _ = pca_with_shift(vecs, 2)

    payload = {
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "prefix": "clustering:",
        "n_words": len(words),
        "n_pcs": N_PCS,
        "explained_variance_ratio_12d": [round(float(x), 4) for x in explained12],
        "explained_total_12d": round(float(explained12.sum()), 4),
        "words": words,                                 # ordered list of word types
        "pca12": [[round(float(x), 4) for x in row] for row in coords12],
        "pca2": [[round(float(x), 4) for x in row] for row in coords2],
    }
    OUT.write_text(json.dumps(payload))
    print(f"Saved {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
