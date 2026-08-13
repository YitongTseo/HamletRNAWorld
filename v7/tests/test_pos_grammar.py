"""Tests for the corpus-derived POS transition grammar (PC11 channel).

Covers the three things that made the OLD learned POS net useless:
  1. it must actually be grammatical (DET->NOUN beats DET->VERB),
  2. it must depend on the history, not just the current word,
  3. it must use the full [0,1] range (the old net lived in 0.05-0.18).
Plus the tag-cache casing bug that was silently mistagging the vocabulary.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import pos_grammar as pg
from server import pos_scorers


def _toy() -> pg.PosGrammar:
    """A tiny grammar over DET NOUN VERB lines, with an EXPLICIT lexicon so the
    test asserts on the transition model rather than on NLTK's guesses."""
    lexicon = {"the": "DET", "a": "DET", "king": "NOUN", "sword": "NOUN",
               "sleeps": "VERB", "falls": "VERB"}
    lines = [["the", "king", "sleeps"]] * 50 + [["a", "sword", "falls"]] * 50
    uni, bi, tri = pg.build_counts(lines, lexicon)
    return pg.PosGrammar(uni, bi, tri, lexicon)


# --- the casing bug -------------------------------------------------------

def test_tag_word_is_case_insensitive():
    """Regression: tag_word cached on word.lower() but tagged the original
    casing, so 'King' (VERB in isolation) could poison the cache slot that
    'king' (NOUN) reads from. Every casing must agree."""
    pos_scorers._pos_cache.clear()
    assert pos_scorers.tag_word("King") == pos_scorers.tag_word("king")
    pos_scorers._pos_cache.clear()
    assert pos_scorers.tag_word("king") == pos_scorers.tag_word("KING")
    pos_scorers._pos_cache.clear()
    assert pos_scorers.tag_word("Love") == pos_scorers.tag_word("love")


def test_tag_word_order_independence():
    """Tagging must not depend on which casing the process saw first."""
    pos_scorers._pos_cache.clear()
    first = [pos_scorers.tag_word(w) for w in ("King", "king", "KING")]
    pos_scorers._pos_cache.clear()
    second = [pos_scorers.tag_word(w) for w in ("KING", "king", "King")]
    assert first == second == [first[0]] * 3


# --- grammaticality -------------------------------------------------------

def test_determiner_prefers_noun_over_verb():
    g = pg.get_grammar()
    after_det = g.context_distribution(["the"])
    assert after_det["NOUN"] > after_det["VERB"]
    assert after_det["NOUN"] > after_det["ADV"]
    # a determiner is the single strongest noun-predictor in English
    assert after_det["NOUN"] == max(after_det.values())


def test_score_depends_on_history():
    """The whole point of the channel: the same candidate word scores
    differently depending on what came before."""
    g = pg.get_grammar()
    assert g.tag("king") == "NOUN"
    # a noun fits far better right after a determiner than right after
    # another noun
    assert g.fit_word("king", ["the"]) > g.fit_word("king", ["sword"])


def test_lexicon_beats_isolated_tagging_on_verbs():
    """The in-context lexicon is the whole reason this module doesn't reuse
    pos_scorers.tag_word: isolated tagging collapses most verbs to NOUN."""
    g = pg.get_grammar()
    verbs = [w for w in ("speak", "think", "know", "give", "make", "hear")
             if w in g._lexicon]
    assert verbs, "expected these common verbs in the play"
    assert all(g.tag(w) == "VERB" for w in verbs)


def test_unknown_word_falls_back_to_isolated_tagger():
    g = pg.get_grammar()
    assert "zzqxwv" not in g._lexicon
    assert g.tag("zzqxwv") in pg.TAGS


def test_toy_corpus_learns_det_noun_verb():
    g = _toy()
    after_det = g.context_distribution(["the"])
    assert after_det["NOUN"] == max(after_det.values())
    after_noun = g.context_distribution(["king"])
    assert after_noun["VERB"] == max(after_noun.values())


# --- dynamic range --------------------------------------------------------

def test_uses_full_range():
    """Max-normalisation means the best continuation is exactly 1.0 and the
    worst is near 0 — unlike the old net, which spanned 0.05-0.18."""
    g = pg.get_grammar()
    for hist in ([], ["the"], ["king"], ["is"], ["the", "in"]):
        dist = g.context_distribution(hist)
        assert abs(max(dist.values()) - 1.0) < 1e-9
        assert min(dist.values()) < 0.1


def test_scores_are_bounded():
    g = pg.get_grammar()
    for hist in ([], ["the"], ["nonexistentwordxyz"], ["", ""]):
        for w in ("king", "the", "sleeps", "nonexistentwordxyz"):
            v = g.fit_word(w, hist)
            assert 0.0 <= v <= 1.0


def test_empty_and_short_history_are_handled():
    g = pg.get_grammar()
    assert 0.0 <= g.fit_word("king", []) <= 1.0
    assert 0.0 <= g.fit_word("king", [""]) <= 1.0
    assert 0.0 <= g.fit_word("king", ["the"]) <= 1.0
    # padding with START must equal an explicitly empty slot
    assert g.fit_word("king", ["the"]) == g.fit_word("king", ["the", ""])


# --- backoff --------------------------------------------------------------

def test_unseen_context_backs_off_not_zero():
    """A trigram context never observed must fall back to the bigram/unigram
    estimate rather than collapsing every tag to 0."""
    uni = Counter({"NOUN": 100, "VERB": 50})
    bi = Counter({("NOUN", "VERB"): 40, ("NOUN", "NOUN"): 10})
    tri = Counter()                       # nothing observed at trigram order
    g = pg.PosGrammar(uni, bi, tri)
    dist = g._distribution("NOUN", "ADJ")  # (ADJ, NOUN) context unseen
    assert dist["VERB"] > dist["NOUN"]     # bigram evidence still drives it
    assert max(dist.values()) == 1.0


def test_determinism():
    g = pg.get_grammar()
    a = [g.fit_word(w, ["the", "in"]) for w in ("king", "sleeps", "the")]
    b = [g.fit_word(w, ["the", "in"]) for w in ("king", "sleeps", "the")]
    assert a == b


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1; print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
