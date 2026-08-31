"""Beowulf corpus — J. Lesslie Hall verse translation, Project Gutenberg
#16328.

Same contract as corpus.hamlet. The poem proper runs between the "I. /
THE LIFE AND DEATH OF SCYLD." heading and "ADDENDA."; everything before
(preface, contents, a long glossary of proper names) and after is cut
entirely — unlike stage directions it is reference apparatus, pages of it,
and would drown the dish in inert words.

Within the poem, inedible (visible but no smell, no eating):
  * {curly-brace glosses} — Hall's prose side-summaries, which can span
    several lines (brace state is tracked line-to-line);
  * fitt headings — roman-numeral lines ("XII.") and all-caps titles;
  * standalone [N] page markers and [N] footnote bodies (footnote text
    starts flush-left with a bracket; verse never does).
Verse lines carry marginal line numbers ("       15 [1]That reaved...")
— digits and brackets are not in the shared tokenizer regex, so they
never become food and need no stripping.
"""
from __future__ import annotations

import re
from pathlib import Path

from corpus.hamlet import tokenize  # shared tokenizer — one regex, one truth

_GUTENBERG_PATH = Path(__file__).parent / "beowulf_gutenberg.txt"

_POEM_START_RE = re.compile(r"^THE LIFE AND DEATH OF SCYLD\.")
_POEM_END_RE = re.compile(r"^ADDENDA\.")
_ROMAN_RE = re.compile(r"^[IVXL]+\.\s*$")
_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z\s'\.,;:\-—]+$")
_BRACKET_LINE_RE = re.compile(r"^\[")  # footnote body or page marker

_CACHE: tuple[list[str], list[bool]] | None = None


def _load() -> tuple[list[str], list[bool]]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    raw = _GUTENBERG_PATH.read_text(encoding="utf-8")
    raw = raw.replace("‘", "'").replace("’", "'")
    raw = raw.replace("“", '"').replace("”", '"')
    all_lines = raw.splitlines()
    start = next((i for i, ln in enumerate(all_lines)
                  if _POEM_START_RE.match(ln.strip())), 0)
    end = next((i for i, ln in enumerate(all_lines)
                if _POEM_END_RE.match(ln.strip())), len(all_lines))
    lines: list[str] = []
    flags: list[bool] = []
    in_gloss = False  # {...} glosses span lines; track brace state
    for rawln in all_lines[start:end]:
        ln = rawln.rstrip()
        if not ln.strip():
            continue
        stripped = ln.strip()
        was_gloss = in_gloss or stripped.startswith("{")
        if "{" in stripped:
            in_gloss = "}" not in stripped[stripped.index("{"):]
        elif in_gloss and "}" in stripped:
            in_gloss = False
        edible = not (
            was_gloss
            or _ROMAN_RE.match(stripped)
            or _ALLCAPS_RE.match(stripped)
            or _BRACKET_LINE_RE.match(stripped)
        )
        lines.append(ln)
        flags.append(edible)
    _CACHE = (lines, flags)
    return _CACHE


def get_sentences_with_flags(passage: str = "full",
                             n: int | None = None) -> tuple[list[str], list[bool]]:
    if passage != "full":
        raise ValueError(f"beowulf corpus has only 'full', got {passage!r}")
    lines, flags = _load()
    if n is not None:
        lines, flags = lines[:n], flags[:n]
    # Token lists per line — the corpus contract (matches hamlet's loader).
    return [tokenize(ln) for ln in lines], list(flags)
