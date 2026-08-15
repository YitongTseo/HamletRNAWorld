"""Corpus registry — the one place that knows which texts exist.

get_sentences_with_flags(corpus, passage) dispatches to the per-text
module; hamlet keeps its passage vocabulary (opening/soliloquy/act1/full),
the others have only "full". TITLES feed the judge prompt and viewer
labels, so a flask's critic and its dish header always name the same text.
"""
from __future__ import annotations

from corpus import beowulf, hamlet, laozi

TITLES = {
    "hamlet": "Shakespeare's Hamlet",
    "laozi": "the Tao Teh King of Lao-Tse (Legge translation)",
    "beowulf": "the Anglo-Saxon epic Beowulf (Hall translation)",
}

_LOADERS = {
    "hamlet": hamlet.get_sentences_with_flags,
    "laozi": laozi.get_sentences_with_flags,
    "beowulf": beowulf.get_sentences_with_flags,
}


def get_sentences_with_flags(corpus: str, passage: str = "full",
                             n: int | None = None):
    try:
        loader = _LOADERS[corpus]
    except KeyError:
        raise ValueError(f"unknown corpus {corpus!r}; have {sorted(_LOADERS)}")
    return loader(passage, n=n) if n is not None else loader(passage)
