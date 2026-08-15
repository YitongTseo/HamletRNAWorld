"""Daodejing corpus — the original Classical Chinese, Project Gutenberg
#7337. The paired experiment to corpus/laozi.py (Legge's English): same
text, two languages, two flasks.

Character-level atoms: Classical Chinese is unsegmented and each character
is a morpheme, so one CJK character = one token = one thing a worm eats.
CJK punctuation (，。；：…) becomes punct tokens — the WORMLET_PUNCT_SMELL
predicate (^[^\\w\\s]+$) already matches it, so marks smell of grammar here
exactly as in English. Latin letters and digits (Gutenberg apparatus,
chapter numerals) never tokenise. PC11 for this corpus is a character
trigram over the text itself (see pos_grammar) — "how expected is this
character after the last two", which is a truer grammar signal than
borrowing an English POS tagger.

Passages: only "full" (~5,700 characters, 810 distinct, 323 lines).
"""
from __future__ import annotations

import re
from pathlib import Path

_GUTENBERG_PATH = Path(__file__).parent / "daodejing_gutenberg.txt"
_START_RE = re.compile(r"\*\*\* START OF.*\*\*\*")
_END_RE = re.compile(r"\*\*\* END OF.*\*\*\*")

# One CJK char, or one CJK punctuation mark, per token.
_TOKEN_RE = re.compile(r"[一-鿿]|[、。，；："
                       r"！？「」『』]")

# Chapter markers ("第一章", bare numerals) and any Latin-heavy apparatus
# line: visible set-dressing, not poem.
_HEADING_RE = re.compile(r"^[\s0-9]*(第?[一二三四五六七八九十百]+章?)?[\s0-9]*$"
                         r"|^[\s\x00-\x7f]+$")

_CACHE: tuple[list[list[str]], list[bool]] | None = None


def _load() -> tuple[list[list[str]], list[bool]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    raw = _GUTENBERG_PATH.read_text(encoding="utf-8")
    m0 = _START_RE.search(raw)
    m1 = _END_RE.search(raw)
    body = raw[m0.end():m1.start()] if (m0 and m1) else raw
    lines: list[list[str]] = []
    flags: list[bool] = []
    for rawln in body.splitlines():
        if not rawln.strip():
            continue
        toks = _TOKEN_RE.findall(rawln)
        if not toks:
            continue  # pure-Latin apparatus lines vanish entirely
        lines.append(toks)
        # Inedible: regex-matched heading lines, plus short lines made only
        # of title/chapter apparatus (老子道德經, 第 N 章 markers inline in
        # this edition — observed leaking into gen-0 poems as eaten 第/章).
        _MARKER = set("第章老子道德經一二三四五六七八九十百")
        apparatus = (len(toks) <= 8 and
                     all(t in _MARKER or not t.isalpha() for t in toks))
        flags.append(not _HEADING_RE.match(rawln.strip()) and not apparatus)
    _CACHE = (lines, flags)
    return _CACHE


def get_sentences_with_flags(passage: str = "full",
                             n: int | None = None) -> tuple[list[list[str]], list[bool]]:
    if passage != "full":
        raise ValueError(f"daodejing corpus has only 'full', got {passage!r}")
    lines, flags = _load()
    if n is not None:
        lines, flags = lines[:n], flags[:n]
    return list(lines), list(flags)
