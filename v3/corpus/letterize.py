"""Align tokens back to character positions in the original sentence.

The simulator wants every *character* (including the space between words) as
its own bead so word lengths physically affect the chain. We still need to
know which token each character belongs to so that the bonding rule can use
*token-level* embedding affinity. This module produces that alignment.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Letterized:
    """Char-level expansion of one sentence aligned to its tokens.

    text:        the original sentence (rendered as-is)
    chars:       list[str] — every character in `text`, in order, length == L
    token_ids:   list[int] of length L. -1 for whitespace / unmatched chars.
    """
    text: str
    chars: list[str]
    token_ids: list[int]

    @property
    def n_letters(self) -> int:
        return len(self.chars)


def letterize(sentence: str, tokens: list[str]) -> Letterized:
    """Walk the original sentence, aligning each character to a token index.

    Tokens must appear in the sentence in order (the typical regex-based
    tokenizer satisfies this). Whitespace between tokens gets token_id = -1
    so it can be marked non-reactive without special casing.
    """
    # Find each token's character span by scanning forward.
    spans: list[tuple[int, int]] = []
    cursor = 0
    for tok in tokens:
        idx = sentence.find(tok, cursor)
        if idx < 0:
            raise ValueError(
                f"token {tok!r} not found in sentence after position {cursor}: "
                f"{sentence!r}"
            )
        spans.append((idx, idx + len(tok)))
        cursor = idx + len(tok)

    chars: list[str] = []
    token_ids: list[int] = []
    pos = 0
    for tok_idx, (start, end) in enumerate(spans):
        # Whitespace / filler before this token.
        for c in sentence[pos:start]:
            chars.append(c)
            token_ids.append(-1)
        # The token's own characters.
        for c in sentence[start:end]:
            chars.append(c)
            token_ids.append(tok_idx)
        pos = end
    # Trailing characters (rare).
    for c in sentence[pos:]:
        chars.append(c)
        token_ids.append(-1)

    return Letterized(text=sentence, chars=chars, token_ids=token_ids)
