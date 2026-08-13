"""POS-tagger-based fitness functions for the four sanity-check experiments.

These replace the Claude LLM judge for the simple-experiment path. They're
deterministic, free, and fast — designed to verify the substrate (UMAP
embeddings + connectome + body + NES) can learn *anything* before trusting
the expensive coherent-poetry runs.

The four modes:

  WORDS     — count edible words eaten (ceiling sanity check; no semantics)
  NOUNS     — count nouns eaten (does the chemosensation steer toward a POS class?)
  ADJ_NOUN  — count adjacent (adj, noun) pairs in the eaten sequence
              (1-step memory: does the worm chain words?)
  POS_CHAIN — count consecutive eaten-word POS bigrams that match a
              curated valid-English set (smoother gradient than a strict
              adj-noun-verb-adv chain)

The POS tagger is NLTK's averaged_perceptron + universal_tagset. Tagging is
done one word at a time against a precomputed corpus-vocab table (built lazily
on first call), so per-generation scoring is O(eaten_words) lookups — no
per-call tagger overhead.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import nltk


# Point NLTK at the project-local data dir if it's there (so deploys don't
# need a system-wide nltk_data install).
_V6_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_NLTK = _V6_ROOT / ".venv" / "nltk_data"
if _LOCAL_NLTK.exists() and str(_LOCAL_NLTK) not in nltk.data.path:
    nltk.data.path.insert(0, str(_LOCAL_NLTK))


# Universal tag set used by NLTK's tagset='universal' mapping. We narrow to
# the open-class tags we actually score on. Penn-tree-bank tags get mapped
# to one of: ADJ, ADP, ADV, CONJ, DET, NOUN, NUM, PRON, PRT, VERB, X, ".".
NOUN, VERB, ADJ, ADV, DET, ADP, PRON, PRT, CONJ = (
    "NOUN", "VERB", "ADJ", "ADV", "DET", "ADP", "PRON", "PRT", "CONJ"
)

# Valid English POS-bigrams for the chain scorer. A pair of adjacent eaten
# words whose tags are in this set contributes 1 to the chain score. Chosen
# to reward natural English chunks (adjectival noun phrases, verbal phrases,
# adverbial modification) while excluding noise like NOUN-NOUN runs that
# look like accidental word-soup. Symmetric where natural — VERB-ADV and
# ADV-VERB both count.
VALID_BIGRAMS: frozenset[tuple[str, str]] = frozenset({
    (ADJ, NOUN),
    (DET, NOUN),
    (DET, ADJ),
    (ADJ, ADJ),     # piled-up adjectives are still grammatical
    (NOUN, VERB),
    (PRON, VERB),
    (VERB, NOUN),
    (VERB, DET),
    (VERB, ADV),
    (ADV, VERB),
    (ADV, ADJ),
    (ADP, NOUN),    # in the garden, by the king
    (ADP, DET),
    (ADP, PRON),
    (PRT, VERB),    # "to be"
    (CONJ, NOUN),
    (CONJ, VERB),
    (CONJ, ADJ),
})


@lru_cache(maxsize=1)
def _pos_for_word(word: str) -> str:
    """Tag a single word, in isolation, using NLTK's perceptron tagger +
    universal mapping. Cached because the eaten-word vocabulary is small
    (~5K unique words across Hamlet) and we'll repeat-tag heavily."""
    # The cache size of 1 above is just to satisfy lru_cache's signature on the
    # outer decorator; the actual cache lives inside _pos_cache below because
    # we want it keyed on lowercased word.
    raise NotImplementedError


_pos_cache: dict[str, str] = {}


def tag_word(word: str) -> str:
    """Return universal-tagset tag for a single word. Lowercased lookup."""
    if not word:
        return "X"
    key = word.lower()
    cached = _pos_cache.get(key)
    if cached is not None:
        return cached
    # NLTK's pos_tag wants a list of tokens; we pass [key] alone.
    # In-isolation tagging is less accurate than in-context tagging but our
    # use case scores eaten sequences that are non-contiguous in the original
    # text, so the per-word tag is the right unit anyway.
    #
    # 2026-08-13 BUGFIX: tag the LOWERCASED form, not the original casing.
    # The cache is keyed on word.lower() but this used to tag `word` as it was
    # written, so a word's tag was decided by whichever capitalisation the
    # process happened to see first — and the perceptron treats a capitalised
    # lone token as a probable proper noun/verb:
    #     pos_tag(["king"]) -> NOUN      pos_tag(["King"]) -> VERB
    #     pos_tag(["love"]) -> NOUN      pos_tag(["Love"]) -> VERB
    # Hamlet is full of sentence-initial and speaker-name capitals, so a large
    # slice of the vocabulary was being mistagged, non-deterministically with
    # respect to corpus order. Lowercasing first makes tagging a pure function
    # of the cache key.
    try:
        tagged = nltk.pos_tag([key], tagset="universal")
        tag = tagged[0][1] if tagged else "X"
    except Exception:
        tag = "X"
    _pos_cache[key] = tag
    return tag


def tag_sequence(words: list[str]) -> list[str]:
    """In-context POS tagging for a full sequence. Used at scoring time so
    the tagger sees neighboring words and resolves NOUN/VERB ambiguity better
    than the per-word path. Falls back to per-word tags on failure."""
    if not words:
        return []
    try:
        tagged = nltk.pos_tag(words, tagset="universal")
        return [t for (_w, t) in tagged]
    except Exception:
        return [tag_word(w) for w in words]


# ----- scorers ----------------------------------------------------------

def score_words(eaten: list[str]) -> float:
    """Raw count of edible words eaten. Punctuation tokens (tagged '.')
    don't count — they're 'eaten' in the corpus sense but aren't words.
    This is the trivial ceiling: any improvement at all means the worm
    learned to chase food."""
    if not eaten:
        return 0.0
    tags = tag_sequence(eaten)
    return float(sum(1 for t in tags if t != "."))


def score_nouns(eaten: list[str]) -> float:
    """Count of NOUN-tagged words eaten. Tests whether selection can push
    the chemosensation toward a single POS class."""
    if not eaten:
        return 0.0
    tags = tag_sequence(eaten)
    return float(sum(1 for t in tags if t == NOUN))


def score_adj_noun(eaten: list[str]) -> float:
    """Count of adjacent (ADJ, NOUN) bigrams in the eaten sequence. The
    minimal memory test — to score, the worm has to chain two semantically
    related words in order, not just eat both at some point."""
    if len(eaten) < 2:
        return 0.0
    tags = tag_sequence(eaten)
    return float(sum(1 for i in range(len(tags) - 1)
                     if tags[i] == ADJ and tags[i + 1] == NOUN))


def score_pos_chain(eaten: list[str]) -> float:
    """Count of adjacent POS bigrams in the eaten sequence that match a
    curated valid-English set (VALID_BIGRAMS). Smooth gradient: every
    well-formed adjacency contributes, and longer well-formed runs
    naturally accumulate more pairs (an N-word valid run yields N-1 pairs).
    This tests how deep a coherent chain the worm can build."""
    if len(eaten) < 2:
        return 0.0
    tags = tag_sequence(eaten)
    return float(sum(1 for i in range(len(tags) - 1)
                     if (tags[i], tags[i + 1]) in VALID_BIGRAMS))


# ----- semantic correctness (LLM-free) ----------------------------------
# The hardest deterministic test: reward eaten sequences whose CONSECUTIVE
# words are semantically related, via cosine similarity of the frozen
# nomic-512 vectors (server/embedding). This asks more than POS chains — not
# just "adjective then noun" but "these two words actually belong near each
# other in meaning space" — while staying free + deterministic.
# Raw nomic vectors sit in a tight cone (random word pairs average ~0.69
# cosine), so raw cosine can't discriminate. We MEAN-CENTER (remove the common
# component) and renormalize, which spreads the distribution out: random pairs
# → ~0.0, genuinely related pairs (king-queen 0.52, death-grave 0.47) stand
# out, and function-word noise (the-and 0.13) is rejected. 0.30 passes strong
# semantic pairs while only ~0.3% of random pairs slip through.
SEM_THRESHOLD = float(os.environ.get("WORMLET_SEM_THRESHOLD", "0.30"))

_SEM_VECS: dict | None = None


def _load_sem_vectors() -> dict:
    """{lowercased_word: np.array(512)} MEAN-CENTERED + L2-renormalized nomic
    vectors. Lazy + imported inside the function to avoid a server.embedding
    <-> pos_scorers import cycle. Empty dict if the cache is missing."""
    global _SEM_VECS
    if _SEM_VECS is None:
        import numpy as np
        from server import embedding  # lazy: breaks the import cycle
        raw = embedding._load_nomic()
        if not raw:
            _SEM_VECS = {}
            return _SEM_VECS
        words = list(raw.keys())
        mean = np.mean(np.stack([raw[w] for w in words]), axis=0)
        centered = {}
        for w in words:
            x = raw[w] - mean
            centered[w] = x / (np.linalg.norm(x) + 1e-12)
        _SEM_VECS = centered
    return _SEM_VECS


def score_semantic(eaten: list[str]) -> float:
    """Count adjacent eaten-word pairs that are DISTINCT and semantically
    close (cosine of their nomic vectors >= SEM_THRESHOLD). OOV tokens
    (punctuation) are transparent — they don't break an otherwise-adjacent
    pair. Repetition earns nothing (identical words excluded), so this can't
    be gamed the way the LLM judge could."""
    if len(eaten) < 2:
        return 0.0
    vecs = _load_sem_vectors()
    if not vecs:
        return 0.0
    import numpy as np
    count = 0
    prev = None
    prev_key = None
    for w in eaten:
        key = w.lower().strip("'").strip()
        v = vecs.get(key)
        if v is None:
            continue                     # OOV / punctuation: transparent
        if prev is not None and key != prev_key:
            if float(np.dot(prev, v)) >= SEM_THRESHOLD:
                count += 1
        prev = v
        prev_key = key
    return float(count)


SCORERS = {
    "words": score_words,
    "nouns": score_nouns,
    "pos_chain": score_pos_chain,
    "semantic": score_semantic,
}


def pos_breakdown(eaten: list[str]) -> dict[str, int]:
    """Frequency of each universal POS tag across the eaten sequence.
    Used by the generations viewer to show what kinds of words a worm is
    actually selecting for, regardless of which scorer is in play."""
    if not eaten:
        return {}
    tags = tag_sequence(eaten)
    out: dict[str, int] = {}
    for t in tags:
        out[t] = out.get(t, 0) + 1
    return out


def longest_valid_chain(eaten: list[str]) -> int:
    """Longest consecutive run of valid bigrams in the eaten sequence.
    Reported alongside score_pos_chain so we can see whether the worm is
    accumulating short hits or genuinely building longer chains."""
    if len(eaten) < 2:
        return 0
    tags = tag_sequence(eaten)
    best = run = 0
    for i in range(len(tags) - 1):
        if (tags[i], tags[i + 1]) in VALID_BIGRAMS:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best + 1 if best > 0 else 0
