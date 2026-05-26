"""Punctuation cleanup applied to worm poems before LLM scoring + git commit.

Rule (per spec):
  - Collapse runs of consecutive punctuation tokens to the FIRST mark.
  - Preserve trailing punctuation (don't strip).

Examples:
  ["he", ".", ".", "."]                              -> ["he", "."]
  ["I", "'tis", "!", "?", ".", "."]                  -> ["I", "'tis", "!"]
  ["to","be",",","or","not","to","be",".",".",".","."] -> ["to","be",",","or","not","to","be","."]

We operate on the token list directly because the worm already emits per-token
output (drain_eaten_words); we never re-tokenize."""
from __future__ import annotations

# Punctuation tokens the worm can encounter via corpus.hamlet's tokenizer.
# Keep in sync with the tokenizer regex in corpus/hamlet.py.
PUNCTUATION: frozenset[str] = frozenset({".", ",", "!", "?", ";", ":", "—", "-"})


def is_punct(tok: str) -> bool:
    return tok in PUNCTUATION


def clean(tokens: list[str]) -> list[str]:
    """Return a new list with runs of consecutive punctuation collapsed to
    the first mark. Trailing punctuation is preserved as-is (a single mark
    at the end stays; a run at the end collapses to its first mark)."""
    out: list[str] = []
    in_run = False
    for tok in tokens:
        if is_punct(tok):
            if in_run:
                # We've already kept the first mark of this run; skip rest.
                continue
            out.append(tok)
            in_run = True
        else:
            out.append(tok)
            in_run = False
    return out
