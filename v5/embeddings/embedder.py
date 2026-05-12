"""DistilBERT word embeddings + PCA, with disk cache."""
from __future__ import annotations
import json
import hashlib
import numpy as np
from pathlib import Path

CACHE = Path(__file__).parent / "cache.json"


def _hash_tokens(tokens: list[str]) -> str:
    """Hash the token list to detect cache invalidation."""
    return hashlib.md5(" ".join(tokens).encode()).hexdigest()[:12]


def embed_and_pca(all_sentences: list[list[str]]) -> dict:
    """Returns {tokens, pca, token_to_idx} for unique word types."""
    # Collect unique tokens, sorted
    flat_tokens = sorted({t for s in all_sentences for t in s})

    # Check cache
    cache_key = _hash_tokens(flat_tokens)
    if CACHE.exists():
        try:
            cached = json.loads(CACHE.read_text())
            if cached.get("key") == cache_key:
                return {k: cached[k] for k in ("tokens", "pca", "token_to_idx")}
        except Exception:
            pass

    print(f"Computing embeddings for {len(flat_tokens)} unique tokens (first run, will cache)...")

    # Import here so transformers/torch aren't required if cache exists
    from transformers import DistilBertTokenizerFast, DistilBertModel
    import torch

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()

    with torch.no_grad():
        enc = tokenizer(
            flat_tokens,
            padding=True,
            truncation=True,
            return_tensors="pt",
            is_split_into_words=False,
        )
        out = model(**enc)
        # CLS token (first token) per word
        vecs = out.last_hidden_state[:, 0, :].numpy()  # (N, 768)

    # PCA to 2D
    vecs_centered = vecs - vecs.mean(axis=0)
    _, _, Vt = np.linalg.svd(vecs_centered, full_matrices=False)
    coords2d = vecs_centered @ Vt[:2].T  # (N, 2)

    # Normalize to [0, 1]
    lo, hi = coords2d.min(axis=0), coords2d.max(axis=0)
    coords2d = (coords2d - lo) / (hi - lo + 1e-9)

    token_to_idx = {t: i for i, t in enumerate(flat_tokens)}
    result = {
        "key": cache_key,
        "tokens": flat_tokens,
        "pca": coords2d.tolist(),
        "token_to_idx": token_to_idx,
    }

    # Write cache
    CACHE.write_text(json.dumps(result))
    print(f"Embeddings cached to {CACHE}")

    return {k: result[k] for k in ("tokens", "pca", "token_to_idx")}
