"""Build the frozen 512-dim nomic embedding cache for v7's LEARNED embedding.

v6 reduced the 512-dim nomic vectors to 12 dims with UMAP and froze that.
v7 keeps the FULL 512-dim vectors frozen and learns the 512->12 mapping with
a small co-evolved net (server/embedding.py). This script produces the frozen
input table those nets consume at runtime.

Pipeline (mirrors build_corpus_umap.py steps 1-2, then stops):
1. Read full Hamlet, strip the Gutenberg envelope, dedupe word types.
2. Embed each type with nomic-embed-text-v1.5 (L2-normalized, "clustering:"
   prefix — SAME as the UMAP build so the semantic space matches).
3. Save raw 512-dim vectors to cache/corpus_nomic512.json.

Keys are lowercased + apostrophe-stripped, matching server/embedding.vec512().

torch + transformers + sentence-transformers live in /home/web/.venv (build
time only; runtime needs only numpy). Run:
    /home/web/.venv/bin/python scripts/build_corpus_nomic512.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

V7_ROOT = Path(__file__).resolve().parent.parent
SRC = V7_ROOT / "corpus" / "hamlet_gutenberg.txt"
OUT = V7_ROOT / "cache" / "corpus_nomic512.json"

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
PREFIX = "clustering:"

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

    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device="cpu")
    inputs = [f"{PREFIX} {w}" for w in words]
    print(f"Embedding {len(inputs)} unique tokens on CPU...")
    t0 = time.monotonic()
    vecs = model.encode(
        inputs, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    print(f"Embedded in {time.monotonic() - t0:.1f}s. Shape: {vecs.shape}")
    return vecs


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not SRC.exists():
        sys.exit(f"missing {SRC}")

    words = extract_unique_words(load_play_text())
    print(f"unique word types: {len(words)}")

    vecs = embed_words(words)
    n_dims = int(vecs.shape[1])

    payload = {
        "model": MODEL_NAME,
        "prefix": PREFIX,
        "n_words": len(words),
        "n_dims": n_dims,
        "words": words,
        "nomic512": [[round(float(x), 6) for x in row] for row in vecs],
    }
    OUT.write_text(json.dumps(payload))
    print(f"Saved {OUT} ({OUT.stat().st_size // 1024} KB), n_dims={n_dims}")
    if n_dims != 512:
        print(f"WARNING: expected 512 dims, got {n_dims}. Update embedding.D_IN.")


if __name__ == "__main__":
    main()
