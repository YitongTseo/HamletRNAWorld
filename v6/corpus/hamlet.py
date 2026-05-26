"""Hamlet text source + simple word/punctuation tokenizer.

The text below is the very opening of Hamlet (Act 1 Scene 1), which gives a
short, dramatic, well-known starting passage. Each sentence becomes one
"strand" in the simulation: extruded into the scene through the same vent,
folding by embedding affinity once it's out.

For the full play, use get_sentences(passage="full") which reads the entire
Gutenberg edition (~5200 lines). One full pass at the default scroll rate
takes ~6 hours and defines one generation in the evolution scheme.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

# Opening of Hamlet, Act 1 Scene 1. Keep this as a list of strings; one per
# sentence. The "sentence" boundary is what defines a strand — a longer or
# different passage just means changing this list.
OPENING_SENTENCES: list[str] = [
    "Who's there?",
    "Nay, answer me.",
    "Stand, and unfold yourself.",
    "Long live the King!",
    "Bernardo?",
    "He.",
    "You come most carefully upon your hour.",
    "'Tis now struck twelve.",
    "Get thee to bed, Francisco.",
    "For this relief much thanks.",
    "'Tis bitter cold, and I am sick at heart.",
    "Have you had quiet guard?",
    "Not a mouse stirring.",
    "Well, good night.",
    "If you do meet Horatio and Marcellus, bid them make haste.",
    "I think I hear them.",
    "Stand, ho!",
    "Who is there?",
    "Friends to this ground.",
    "And liegemen to the Dane.",
    "Give you good night.",
    "O, farewell, honest soldier.",
    "Who hath relieved you?",
    "Bernardo hath my place.",
    "Holla! Bernardo!",
    "Say, what, is Horatio there?",
    "A piece of him.",
    "Welcome, Horatio.",
    "Welcome, good Marcellus.",
    "What, has this thing appeared again tonight?",
    "I have seen nothing.",
    "Horatio says 'tis but our fantasy.",
    "And will not let belief take hold of him.",
    "Touching this dreaded sight, twice seen of us.",
    "Therefore I have entreated him along.",
    "With us to watch the minutes of this night.",
    "That if again this apparition come,",
    "He may approve our eyes and speak to it.",
    "Tush, tush, 'twill not appear.",
    "Sit down awhile.",
    "And let us once again assail your ears.",
    "That are so fortified against our story.",
    "What we two nights have seen.",
    "Well, sit we down.",
    "And let us hear Bernardo speak of this.",
    "Last night of all,",
    "When yon same star that's westward from the pole,",
    "Had made his course t'illume that part of heaven,",
    "Where now it burns,",
    "Marcellus and myself, the bell then beating one,",
    "Peace, break thee off.",
    "Look where it comes again!",
]


# A common, more iconic passage — uncomment via get_sentences(passage="soliloquy").
TO_BE_SOLILOQUY: list[str] = [
    "To be, or not to be, that is the question.",
    "Whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune.",
    "Or to take arms against a sea of troubles, and by opposing end them.",
]


_PASSAGES = {
    "opening": OPENING_SENTENCES,
    "soliloquy": TO_BE_SOLILOQUY,
}

# The full play lives in the Gutenberg dump. Loaded lazily because it's ~200KB
# and most callers only want the small opening passage. Curly typographic
# quotes get normalized to straight ASCII so the tokenizer regex below matches
# contractions like "appear'd".
_GUTENBERG_PATH = Path(__file__).resolve().parent / "hamlet_gutenberg.txt"
_GUTENBERG_START_RE = re.compile(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
_GUTENBERG_END_RE = re.compile(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
_FULL_PLAY_CACHE: list[str] | None = None


def _load_full_play() -> list[str]:
    """Return the full play as a list of raw line strings (one per line of
    dialogue/stage direction in the Gutenberg edition). Empty lines stripped."""
    global _FULL_PLAY_CACHE
    if _FULL_PLAY_CACHE is not None:
        return _FULL_PLAY_CACHE
    raw = _GUTENBERG_PATH.read_text(encoding="utf-8")
    start = _GUTENBERG_START_RE.search(raw)
    end = _GUTENBERG_END_RE.search(raw)
    body = raw[start.end():end.start()] if (start and end) else raw
    # Normalize curly quotes → straight so contractions like "appear'd" tokenize.
    body = body.replace("’", "'").replace("‘", "'")
    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
    _FULL_PLAY_CACHE = lines
    return lines


# Filler / non-reactive tokens. These pass through the simulation as inert
# beads — they have positions, get pushed around by the chain, but never
# form pair-bonds. Mirrors how stop-word lists work in NLP, plus all
# punctuation (which we don't want bonding to content words).
NON_REACTIVE: frozenset[str] = frozenset({
    # English stop / function words.
    "a", "an", "the",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "us", "them",
    "my", "your", "his", "her", "our", "their",
    "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "am", "been", "being",
    "do", "does", "did",
    "have", "has", "had",
    "and", "or", "but", "nor", "so", "yet",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "as",
    "if", "then", "than",
    # Punctuation.
    ",", ".", "!", "?", ";", ":", "—", "-", "'",
})


def is_non_reactive(token: str) -> bool:
    """True for stop words + punctuation that shouldn't form pair-bonds."""
    return token.lower() in NON_REACTIVE


# Tokenizer: split on whitespace, but pull punctuation off as its own tokens.
# Keeps apostrophes inside contractions ("Who's", "'Tis") since BERT-family
# tokenizers handle them sensibly and they're meaningful as words.
_TOKEN_RE = re.compile(r"[A-Za-z']+|[.,!?;:—\-]")


def tokenize(sentence: str) -> list[str]:
    """Word-and-punctuation tokenization. Empty strings filtered out."""
    return [t for t in _TOKEN_RE.findall(sentence) if t.strip()]


def _get_raw_passage(passage: str) -> list[str]:
    if passage == "full":
        return _load_full_play()
    if passage in _PASSAGES:
        return _PASSAGES[passage]
    raise ValueError(f"unknown passage {passage!r}; pick from {list(_PASSAGES) + ['full']}")


def get_sentences(passage: str = "opening", n: int | None = None) -> list[list[str]]:
    """Return tokenized sentences from the chosen passage."""
    raw = _get_raw_passage(passage)
    if n is not None:
        raw = raw[:n]
    return [tokenize(s) for s in raw]


def get_raw_and_tokens(
    passage: str = "opening", n: int | None = None
) -> tuple[list[str], list[list[str]]]:
    """Return (raw_sentence_strings, tokenized_sentences) — letter-level work
    needs both the original surface form (for rendering) and the tokens (for
    embeddings)."""
    raw = _get_raw_passage(passage)
    if n is not None:
        raw = raw[:n]
    return list(raw), [tokenize(s) for s in raw]


def flatten(sentences: Iterable[list[str]]) -> list[str]:
    """Concatenate all words across all sentences in order."""
    out: list[str] = []
    for s in sentences:
        out.extend(s)
    return out
