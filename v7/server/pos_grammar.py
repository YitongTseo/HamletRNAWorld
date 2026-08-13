"""Corpus-derived POS transition grammar — the PC11 chemosensory channel.

WHAT THIS ANSWERS
    "How well does this candidate word fit, grammatically, given the words the
    worm has already eaten?"

The worm's eaten sequence is its own emergent output: a non-contiguous walk
through Hamlet's vocabulary. PC11 scores each *candidate* word in smelling
range by how plausible its part of speech is as a continuation of the POS
sequence the worm has already swallowed. A worm that steers up this gradient
is assembling grammatical English.

WHY THIS REPLACED THE LEARNED POS NET (2026-08-13)
    v7 computed PC11 with a randomly-initialised 72->16->1 net evolved by the
    per-flask (1+1)-ES. Two things were wrong with it. First, the ES accepted
    2-7 mutations in 101 generations (zero in the last 30 for most flasks), so
    the net never left its random initialisation — it computed an arbitrary
    function of the POS sequence, not a grammatical one, and it happened to
    rank NOUN *lowest*. Second, its output occupied ~0.05-0.18 of the [0,1]
    range, roughly a sixth of the amplitude of the other channels.

    This module is deterministic, costs nothing to evaluate, needs no training,
    and uses the full [0,1] range by construction.

THE MODEL
    An interpolated trigram over universal POS tags, estimated from the Hamlet
    corpus itself, so what counts as "grammatical" is Shakespeare's own syntax
    rather than a hand-written rule table:

        P(t | p1, p2) = L3*P3(t|p1,p2) + (1-L3)*[L2*P2(t|p1) + (1-L2)*P1(t)]

    with Jelinek-Mercer confidence weights L = n/(n + SMOOTHING), where n is
    how often that context was observed. Sparse contexts back off smoothly to
    the bigram and then to the unigram distribution — no cliffs.

    The returned score is then max-normalised within the context:

        fit(t | ctx) = P(t | ctx) / max_t' P(t' | ctx)

    so the single most grammatical continuation scores exactly 1.0 and an
    ungrammatical one scores near 0. That normalisation is what buys the full
    dynamic range: an un-normalised probability would sit near 1/12 for
    everything and reintroduce exactly the flat-signal problem we just fixed.

TAGGING (load-bearing)
    Whatever tagger this uses, the transition table and the runtime queries
    MUST use the same one — otherwise the table estimates probabilities over a
    different tag distribution than the one being queried.

    Tagging each word in isolation (`pos_scorers.tag_word`) satisfies that, but
    is very weak: NLTK's perceptron sees a lone token with no neighbours and
    falls back on its prior, which collapses 3513 of Hamlet's 4563 word types
    (77%) to NOUN — "sleeps", "falls" and "runs" all come back NOUN. A grammar
    over those tags is mostly a grammar of one tag.

    So we build our own lexicon instead: tag every dialogue line of the play
    IN CONTEXT (where the perceptron is accurate), then take each word type's
    MAJORITY tag across all its occurrences. That is a fixed word->tag table,
    so runtime lookup is still O(1) and still consistent with the table, but
    the tags are the ones a real tagger assigns in real sentences. Words never
    seen in the play fall back to the isolated tagger.

    This lexicon is deliberately local to this module — `pos_scorers.tag_word`
    keeps its existing isolated-tagging semantics because the sanity-experiment
    scorers (nouns, pos_chain) are mid-flight against it.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from server import pos_scorers

V7_ROOT = Path(__file__).resolve().parent.parent
_CACHE = V7_ROOT / "cache" / "pos_transitions.json"

# Jelinek-Mercer smoothing constant. A context seen SMOOTHING times gets half
# its weight from the higher-order estimate and half from the backoff. 5 is
# low enough that common contexts (DET->, NOUN->) are essentially pure
# trigram/bigram, high enough that a context seen once doesn't dominate.
SMOOTHING = 5.0

# Tag inventory — universal tagset plus "X" for unknown and "." for
# punctuation, matching what pos_scorers.tag_word can return.
TAGS: list[str] = ["NOUN", "VERB", "ADJ", "ADV", "DET", "ADP",
                   "PRON", "PRT", "CONJ", "NUM", "X", "."]

# The tag used for "no previous word" — a worm that has eaten nothing yet, or
# a padding slot. Distinct from "X" (a real word we failed to tag) so that
# sequence-start gets its own transition statistics.
START = "<S>"


class PosGrammar:
    """Interpolated POS trigram with max-normalised scoring."""

    def __init__(self, uni: Counter, bi: Counter, tri: Counter,
                 lexicon: dict[str, str] | None = None):
        self._lexicon = lexicon or {}
        self._uni = uni
        self._bi = bi          # (p1, t) -> count
        self._tri = tri        # (p2, p1, t) -> count
        self._uni_total = sum(uni.values()) or 1
        # Context marginals, so we never re-sum in the hot path.
        self._bi_ctx: Counter = Counter()
        for (p1, _t), c in bi.items():
            self._bi_ctx[p1] += c
        self._tri_ctx: Counter = Counter()
        for (p2, p1, _t), c in tri.items():
            self._tri_ctx[(p2, p1)] += c
        # fit() is called once per in-range word per brain tick; the number of
        # distinct (p2,p1) contexts is tiny (~170), so memoising the whole
        # normalised distribution per context makes it a dict lookup.
        self._dist_cache: dict[tuple[str, str], dict[str, float]] = {}

    # ---- estimation ----
    def _distribution(self, p1: str, p2: str) -> dict[str, float]:
        """Max-normalised fit score for every tag given the two-tag context."""
        key = (p2, p1)
        hit = self._dist_cache.get(key)
        if hit is not None:
            return hit

        n_tri = self._tri_ctx.get(key, 0)
        n_bi = self._bi_ctx.get(p1, 0)
        l3 = n_tri / (n_tri + SMOOTHING)
        l2 = n_bi / (n_bi + SMOOTHING)

        probs: dict[str, float] = {}
        for t in TAGS:
            p1_uni = self._uni.get(t, 0) / self._uni_total
            p2_bi = (self._bi.get((p1, t), 0) / n_bi) if n_bi else 0.0
            p3_tri = (self._tri.get((p2, p1, t), 0) / n_tri) if n_tri else 0.0
            backoff = l2 * p2_bi + (1.0 - l2) * p1_uni
            probs[t] = l3 * p3_tri + (1.0 - l3) * backoff

        peak = max(probs.values()) or 1.0
        dist = {t: p / peak for t, p in probs.items()}
        self._dist_cache[key] = dist
        return dist

    # ---- tagging ----
    def tag(self, word: str) -> str:
        """This grammar's tag for a word: the majority in-context tag from the
        play, falling back to isolated tagging for out-of-play words."""
        if not word:
            return START
        key = word.lower()
        hit = self._lexicon.get(key)
        return hit if hit is not None else pos_scorers.tag_word(key)

    # ---- public API ----
    def fit_tag(self, tag: str, history_tags: list[str]) -> float:
        """Grammatical fit in [0,1] of `tag` following `history_tags`.

        history_tags is MOST-RECENT-FIRST (matching the worm's _recent_eaten),
        so history_tags[0] is the last word eaten. Missing/short history is
        padded with START."""
        p1 = history_tags[0] if len(history_tags) > 0 and history_tags[0] else START
        p2 = history_tags[1] if len(history_tags) > 1 and history_tags[1] else START
        return self._distribution(p1, p2).get(tag, 0.0)

    def fit_word(self, word: str, history_words: list[str]) -> float:
        """Same, but tags the words for you. history_words most-recent-first."""
        return self.fit_tag(self.tag(word), [self.tag(w) for w in history_words])

    def context_distribution(self, history_words: list[str]) -> dict[str, float]:
        """The whole normalised tag->fit map for a history. Used by the batched
        embedding path, which shares one history across many candidate words:
        look this up once per tick, then index it per candidate word."""
        p1 = self.tag(history_words[0]) if history_words else START
        p2 = self.tag(history_words[1]) if len(history_words) > 1 else START
        return self._distribution(p1, p2)


# --- table construction ---------------------------------------------------

def build_lexicon(lines: list[list[str]]) -> dict[str, str]:
    """word (lowercased) -> its MAJORITY in-context tag across the play.

    Each line is tagged as a sequence, so the perceptron gets the neighbouring
    words it needs to tell a verb from a noun. Ties break toward the tag seen
    first, which `Counter.most_common` already does deterministically."""
    votes: dict[str, Counter] = {}
    for line in lines:
        for word, tag in zip(line, pos_scorers.tag_sequence(line)):
            votes.setdefault(word.lower(), Counter())[tag] += 1
    return {w: c.most_common(1)[0][0] for w, c in votes.items()}


def build_counts(lines: list[list[str]],
                 lexicon: dict[str, str]) -> tuple[Counter, Counter, Counter]:
    """Count uni/bi/trigrams of lexicon tags, one line at a time.

    The context resets to START at each line boundary so we never learn a
    transition that spans two unrelated speeches."""
    uni: Counter = Counter()
    bi: Counter = Counter()
    tri: Counter = Counter()
    for line in lines:
        tags = [lexicon.get(t.lower()) or pos_scorers.tag_word(t) for t in line]
        uni.update(tags)
        prev1 = START
        prev2 = START
        for t in tags:
            bi[(prev1, t)] += 1
            tri[(prev2, prev1, t)] += 1
            prev2, prev1 = prev1, t
    return uni, bi, tri


def _corpus_tokens() -> list[list[str]]:
    """Dialogue lines of the full play, tokenised, one list per line.

    DIALOGUE ONLY. `get_sentences` would also hand back speaker-name lines
    ("KING."), act/scene cues and stage directions — none of which are English
    grammar, and none of which the worm can eat or smell anyway (the sim marks
    them inedible via the same `is_dialogue_line` flag). Letting them into the
    table would teach the grammar that a NOUN frequently follows a full stop
    at the start of a line, which is an artefact of the typesetting.

    Kept as a list-of-lines rather than one flat stream so transitions don't
    run across a line boundary into an unrelated speech."""
    from corpus import hamlet
    sentences, edible = hamlet.get_sentences_with_flags(passage="full")
    return [s for s, ok in zip(sentences, edible) if ok and s]


def build_and_cache(path: Path = _CACHE) -> "PosGrammar":
    """Build the lexicon + transition table from the Hamlet corpus, cache it."""
    lines = _corpus_tokens()
    lexicon = build_lexicon(lines)
    uni, bi, tri = build_counts(lines, lexicon)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "smoothing": SMOOTHING,
        "lexicon": lexicon,
        "uni": dict(uni),
        # JSON keys must be strings; join with a character that can't appear
        # in a universal tag.
        "bi": {"|".join(k): v for k, v in bi.items()},
        "tri": {"|".join(k): v for k, v in tri.items()},
    }))
    return PosGrammar(uni, bi, tri, lexicon)


_grammar: PosGrammar | None = None


def get_grammar() -> PosGrammar:
    """Process-wide singleton. Loads the cached table, building it on first
    use if the cache is absent."""
    global _grammar
    if _grammar is not None:
        return _grammar
    if _CACHE.exists():
        raw = json.loads(_CACHE.read_text())
        uni = Counter(raw["uni"])
        bi = Counter({tuple(k.split("|")): v for k, v in raw["bi"].items()})
        tri = Counter({tuple(k.split("|")): v for k, v in raw["tri"].items()})
        _grammar = PosGrammar(uni, bi, tri, raw.get("lexicon", {}))
    else:
        _grammar = build_and_cache()
    return _grammar


if __name__ == "__main__":
    g = build_and_cache()
    print(f"wrote {_CACHE}  ({len(g._lexicon)} word types in lexicon)")
    counts = Counter(g._lexicon.values())
    print("lexicon tag distribution: "
          + "  ".join(f"{t}={c}" for t, c in counts.most_common()))
    print("\nfit scores by context (most-recent-first history):")
    for hist in ([], ["the"], ["the", "in"], ["king"], ["sleeps"], ["and"]):
        dist = g.context_distribution(hist)
        top = sorted(dist.items(), key=lambda kv: -kv[1])[:5]
        label = " ".join(reversed(hist)) or "<start>"
        tags = "/".join(g.tag(w) for w in hist) or "-"
        print(f"  after {label!r:14} [{tags:<9}] "
              + "  ".join(f"{t}={v:.2f}" for t, v in top))
