"""
Download and parse Hamlet into RNA-strand sentences.

Each sentence is a list of Token namedtuples:
    Token(word: str, is_filler: bool)

Only alphanumeric words are kept; punctuation is stripped.
"""

import re
import os
import pickle
import requests
from dataclasses import dataclass
from typing import List

from config import FILLER_WORDS, TEST_STRANDS

HAMLET_URL = (
    "https://www.gutenberg.org/files/1524/1524-0.txt"
)
HAMLET_CACHE = "cache/hamlet.txt"

# Regex: one or more word characters (letters, digits, apostrophes kept)
_WORD_RE = re.compile(r"[A-Za-z']+")

# Sentence splitter: split on . ! ? followed by whitespace or end-of-string
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Token:
    word: str         # lower-case
    raw:  str         # original capitalisation
    is_filler: bool


def _fetch_hamlet() -> str:
    os.makedirs("cache", exist_ok=True)
    if os.path.exists(HAMLET_CACHE):
        with open(HAMLET_CACHE, "r", encoding="utf-8") as f:
            return f.read()

    print("Downloading Hamlet from Project Gutenberg…")
    r = requests.get(HAMLET_URL, timeout=30)
    r.raise_for_status()
    text = r.text

    # Strip Gutenberg header/footer
    start = text.find("THE TRAGEDY OF HAMLET")
    end   = text.rfind("End of the Project Gutenberg")
    if start != -1:
        text = text[start: end if end != -1 else len(text)]

    with open(HAMLET_CACHE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved to {HAMLET_CACHE}  ({len(text):,} chars)")
    return text


def _split_sentences(text: str) -> List[str]:
    """Very lightweight sentence splitter — good enough for Shakespeare."""
    # Collapse stage directions / ACT/SCENE headers (all-caps lines)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Skip lines that are ALL CAPS (stage directions / headings)
        if stripped.isupper() and len(stripped) > 2:
            continue
        lines.append(stripped)
    blob = " ".join(lines)

    # Split on sentence-ending punctuation
    raw_sents = _SENT_RE.split(blob)
    sents = []
    for s in raw_sents:
        s = s.strip()
        if s:
            sents.append(s)
    return sents


def _tokenise(sentence: str) -> List[Token]:
    tokens = []
    for raw in _WORD_RE.findall(sentence):
        word = raw.lower().rstrip("'").lstrip("'")
        if not word:
            continue
        tokens.append(Token(
            word=word,
            raw=raw,
            is_filler=(word in FILLER_WORDS),
        ))
    return tokens


def load_strands(min_len: int = 4, max_len: int = 80) -> List[List[Token]]:
    """
    Return list of token-lists, one per usable sentence.
    Sentences shorter than min_len words or with no semantic (non-filler)
    words are skipped.
    """
    os.makedirs("cache", exist_ok=True)
    strand_cache = "cache/strands.pkl"
    if os.path.exists(strand_cache):
        with open(strand_cache, "rb") as f:
            strands = pickle.load(f)
        print(f"Loaded {len(strands)} strands from cache.")
        return strands

    text = _fetch_hamlet()
    sentences = _split_sentences(text)
    strands = []
    for sent in sentences:
        tokens = _tokenise(sent)
        if len(tokens) < min_len:
            continue
        semantic_count = sum(1 for t in tokens if not t.is_filler)
        if semantic_count < 2:
            continue
        strands.append(tokens[:max_len])

    with open(strand_cache, "wb") as f:
        pickle.dump(strands, f)
    print(f"Parsed {len(strands)} strands from {len(sentences)} sentences.")
    return strands


def load_test_strands() -> List[List[Token]]:
    """
    Return Token lists directly from TEST_STRANDS — all semantic, no fillers.
    """
    strands = []
    for word_list in TEST_STRANDS:
        tokens = [
            Token(word=w.lower(), raw=w, is_filler=False)
            for w in word_list
            if w.strip()
        ]
        if tokens:
            strands.append(tokens)
    print(f"Loaded {len(strands)} test strands "
          f"({sum(len(s) for s in strands)} total words).")
    return strands


def unique_semantic_words(strands: List[List[Token]]) -> List[str]:
    """Return sorted list of unique non-filler words across all strands."""
    words = set()
    for strand in strands:
        for tok in strand:
            if not tok.is_filler:
                words.add(tok.word)
    return sorted(words)


if __name__ == "__main__":
    strands = load_strands()
    print(f"\nFirst 3 strands:")
    for s in strands[:3]:
        print("  ", " ".join(t.word for t in s))
    print(f"\nUnique semantic words: {len(unique_semantic_words(strands))}")
