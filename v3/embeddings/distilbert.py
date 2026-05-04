"""Contextual word embeddings via DistilBERT.

Each token's vector is the mean of the model's last-layer hidden states for
the subword tokens that span it. CLS/SEP and other special tokens are
ignored. The embedding is **contextual** — the same surface word in two
different sentences will (correctly) get two different vectors.

Swap to any other HuggingFace masked-LM by passing model_name=...
"""
from __future__ import annotations

import numpy as np


class DistilBertEmbedder:
    """Wraps a HuggingFace AutoModel to produce per-word contextual vectors."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: str | None = None,
    ):
        # Lazy heavy imports so users who pick a different embedder don't pay
        # the torch / transformers import cost.
        from transformers import AutoModel, AutoTokenizer
        import torch

        self._torch = torch
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self.device = device
        self.model.to(device)
        self._dim = int(self.model.config.hidden_size)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self.model_name

    # ------------------------------------------------------------------

    def embed_sentence(self, words: list[str]) -> np.ndarray:
        return self.embed_sentences([words])[0]

    def embed_sentences(self, sentences: list[list[str]]) -> list[np.ndarray]:
        if not sentences:
            return []
        torch = self._torch
        encoded = self.tokenizer(
            sentences,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.inference_mode():
            output = self.model(**encoded)
        # (batch, seq_len, dim)
        hidden = output.last_hidden_state.cpu().numpy()

        results: list[np.ndarray] = []
        for b, words in enumerate(sentences):
            # word_ids() needs the BatchEncoding from the tokenizer call; we
            # re-tokenize this one row to get its word-id alignment.
            single = self.tokenizer(
                words,
                is_split_into_words=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
            word_ids = single.word_ids(0)
            seq = hidden[b]                                  # (seq_len, dim)
            n = len(words)
            sums = np.zeros((n, self._dim), dtype=np.float32)
            counts = np.zeros(n, dtype=np.int32)
            for tok_idx, w_id in enumerate(word_ids):
                if w_id is None or tok_idx >= seq.shape[0]:
                    continue
                sums[w_id] += seq[tok_idx]
                counts[w_id] += 1
            counts = counts.clip(min=1)
            results.append(sums / counts[:, None])
        return results
