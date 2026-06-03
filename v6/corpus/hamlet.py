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
# Match "ACT I" alone (not "ACT II"/"III"/...) — used to find Act 1's start.
_ACT_I_RE = re.compile(r"^ACT\s+I\s*$", re.IGNORECASE)
_ACT_II_RE = re.compile(r"^ACT\s+II\s*$", re.IGNORECASE)
_ACT1_CACHE: list[str] | None = None


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


# --- Dialogue vs. set-dressing classification -------------------------------
# Only spoken dialogue is edible / chemosensory. Speaker-name headings
# ("BARNARDO."), act/scene cues ("SCENE I. Elsinore..."), and stage
# directions ("Enter Francisco and Barnardo, two sentinels." / "[_Exit._]")
# are inert decoration: they still scroll past and render, but the worm
# can't eat them and they emit no smell.
_SCENE_RE = re.compile(r"^(ACT|SCENE)\b", re.IGNORECASE)
# Lines that open with an entrance/exit/effect verb are bare stage directions
# in this Gutenberg edition (mid-line directions are bracketed/italicized).
_STAGE_VERB_RE = re.compile(
    r"^(Enter|Exit|Exeunt|Re-enter|Manet|Manent|Flourish|Alarums?|"
    r"Hautboys|Trumpets?|Drums?)\b"
)


def is_dialogue_line(raw_line: str) -> bool:
    """True only for spoken dialogue. False for speaker-name headings,
    ACT/SCENE cues, and stage directions — the inert set-dressing the worm
    must neither eat nor smell."""
    s = raw_line.replace("﻿", "").strip()
    if not s:
        return False
    # Bracketed or fully-italicized stage direction: [_Exit._], _Exeunt._
    if (s[0] == "[" and s[-1] == "]") or (s[0] == "_" and s[-1] == "_"):
        return False
    # ACT / SCENE heading.
    if _SCENE_RE.match(s):
        return False
    # All-caps speaker name (no lowercase letters but at least one letter):
    # "BARNARDO.", "KING CLAUDIUS.", "GHOST.". Dialogue always lives on its
    # own line in this edition, so this never swallows spoken text.
    if any(c.isalpha() for c in s) and not any(c.islower() for c in s):
        return False
    # Bare stage direction opening with an entrance/exit/effect verb.
    if _STAGE_VERB_RE.match(s):
        return False
    return True


def _load_act1() -> list[str]:
    """Just Act 1, sliced from the full Gutenberg edition. ~1200 lines,
    short enough for the sanity-check experiments to iterate fast.

    The Gutenberg dump opens with a table of contents that lists ACT I..V
    one after another, so we take the *second* occurrence of each marker
    (the first is the TOC, the second is the actual play heading)."""
    global _ACT1_CACHE
    if _ACT1_CACHE is not None:
        return _ACT1_CACHE
    lines = _load_full_play()
    acti_idxs = [i for i, ln in enumerate(lines) if _ACT_I_RE.match(ln)]
    actii_idxs = [i for i, ln in enumerate(lines) if _ACT_II_RE.match(ln)]
    if len(acti_idxs) >= 2 and len(actii_idxs) >= 2:
        start, end = acti_idxs[1], actii_idxs[1]
    elif acti_idxs and actii_idxs:
        start, end = acti_idxs[0], actii_idxs[0]
    else:
        # Defensive fallback: just use the first ~1500 lines.
        _ACT1_CACHE = lines[:1500]
        return _ACT1_CACHE
    _ACT1_CACHE = lines[start:end]
    return _ACT1_CACHE


def _get_raw_passage(passage: str) -> list[str]:
    if passage == "full":
        return _load_full_play()
    if passage == "act1":
        return _load_act1()
    if passage in _PASSAGES:
        return _PASSAGES[passage]
    raise ValueError(f"unknown passage {passage!r}; pick from {list(_PASSAGES) + ['act1', 'full']}")


def get_sentences(passage: str = "opening", n: int | None = None) -> list[list[str]]:
    """Return tokenized sentences from the chosen passage."""
    raw = _get_raw_passage(passage)
    if n is not None:
        raw = raw[:n]
    return [tokenize(s) for s in raw]


def get_sentences_with_flags(
    passage: str = "opening", n: int | None = None
) -> tuple[list[list[str]], list[bool]]:
    """Like get_sentences, but also return a parallel list of per-sentence
    `edible` flags. False marks set-dressing (speaker names, ACT/SCENE cues,
    stage directions) the worm must not eat or smell; True marks dialogue."""
    raw = _get_raw_passage(passage)
    if n is not None:
        raw = raw[:n]
    return [tokenize(s) for s in raw], [is_dialogue_line(s) for s in raw]


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
