"""Tao Teh King corpus — Legge translation, Project Gutenberg #216.

Same contract as corpus.hamlet: get_sentences_with_flags() returns one
scroller strand per source line plus a parallel edible list. This text is
almost entirely poem: the only inedible lines are the PART headings and
the title/translator front matter. Chapter/verse numbers ("Ch. 1. 1.")
ride inside edible lines but never become food — the shared tokenizer
regex ([A-Za-z']+ | punctuation) has no digit class, so they vanish at
tokenisation rather than needing a predicate.

Passages: only "full" (~1,000 non-blank lines — close to Hamlet act1, so
generation lengths are comparable without pacing changes).
"""
from __future__ import annotations

import re
from pathlib import Path

from corpus.hamlet import tokenize  # shared tokenizer — one regex, one truth

_GUTENBERG_PATH = Path(__file__).parent / "laozi_gutenberg.txt"
_START_RE = re.compile(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*\*\*\*")
_END_RE = re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*\*\*\*")

# Front matter and structure markers — visible set-dressing, not food.
_INEDIBLE_RE = re.compile(
    r"^\s*(PART\s+[\dIVX]+|THE TAO TEH KING|OR\s*$|THE TAO AND ITS|by Lao-Tse|"
    r"Translated by)", re.IGNORECASE)

_BODY_CACHE: list[str] | None = None


def _load_body() -> list[str]:
    global _BODY_CACHE
    if _BODY_CACHE is None:
        raw = _GUTENBERG_PATH.read_text(encoding="utf-8")
        m0 = _START_RE.search(raw)
        m1 = _END_RE.search(raw)
        body = raw[m0.end():m1.start()] if (m0 and m1) else raw
        body = body.replace("‘", "'").replace("’", "'")
        body = body.replace("“", '"').replace("”", '"')
        _BODY_CACHE = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
    return _BODY_CACHE


def is_poem_line(raw_line: str) -> bool:
    return not _INEDIBLE_RE.match(raw_line)


def get_sentences_with_flags(passage: str = "full",
                             n: int | None = None) -> tuple[list[str], list[bool]]:
    if passage != "full":
        raise ValueError(f"laozi corpus has only 'full', got {passage!r}")
    lines = _load_body()
    if n is not None:
        lines = lines[:n]
    # Token lists per line — the corpus contract (matches hamlet's loader;
    # TextScroller and the POS trainer both consume tokens, not strings).
    return [tokenize(ln) for ln in lines], [is_poem_line(ln) for ln in lines]
