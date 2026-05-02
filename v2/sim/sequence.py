"""RNA bases, Watson-Crick rules, and the test sequence."""
from __future__ import annotations

import numpy as np

# Base encoding: A=0, U=1, G=2, C=3.
A, U, G, C = 0, 1, 2, 3
BASE_CHARS = "AUGC"
BASE_INDEX = {ch: i for i, ch in enumerate(BASE_CHARS)}

# Watson-Crick pairs only (no G-U wobble for v0). Symmetric.
_PAIR_TABLE = np.zeros((4, 4), dtype=bool)
_PAIR_TABLE[A, U] = _PAIR_TABLE[U, A] = True
_PAIR_TABLE[G, C] = _PAIR_TABLE[C, G] = True


def encode(seq: str) -> np.ndarray:
    """Encode an RNA string into a (N,) int array of base indices."""
    seq = seq.upper().replace("T", "U").replace(" ", "")
    return np.array([BASE_INDEX[ch] for ch in seq], dtype=np.int8)


def decode(bases: np.ndarray) -> str:
    return "".join(BASE_CHARS[b] for b in bases)


def can_pair(b1: int, b2: int) -> bool:
    return bool(_PAIR_TABLE[b1, b2])


def pair_table() -> np.ndarray:
    """Return the 4x4 boolean pairing matrix (vectorized lookups)."""
    return _PAIR_TABLE


# 50-nt test sequence: pure-G arm + GAAA tetraloop + pure-C arm.
# Stem 1 is all G's, stem 2 is all C's — neither can self-pair (G-G and C-C are
# not Watson-Crick). The only available pairings are between stems, so this
# forces a clean cross-strand hairpin and gives us a clean visual verification.
HAIRPIN_50 = (
    "GGGGGGGGGGGGGGGGGGGGGGG"  # stem 1 (positions 0-22, all G)
    "GAAA"                      # loop  (positions 23-26)
    "CCCCCCCCCCCCCCCCCCCCCCC"  # stem 2 (positions 27-49, all C)
)
assert len(HAIRPIN_50) == 50
