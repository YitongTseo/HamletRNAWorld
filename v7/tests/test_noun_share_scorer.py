"""Tests for score_noun_share — the volume-proof version of score_nouns.

Added 2026-09-05 alongside the scorer. The property that matters is the one
score_nouns lacks: eating MORE at an unchanged noun share must not raise the
score. The `all` vs `chemo` control spent 1,071 generations demonstrating
that score_nouns rewards exactly that (words/life 28.9 -> 175.6, noun share
23.6% -> 23.1%), so these tests pin the fix.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import pos_scorers as ps
from corpus.hamlet import get_sentences_with_flags


def _act1_tokens():
    sents, edible = get_sentences_with_flags("act1")
    return [w for s, ok in zip(sents, edible) if ok for w in s]


def test_volume_at_constant_share_does_not_pay():
    """The whole point. Duplicating a poem leaves the share untouched, where
    score_nouns doubles."""
    toks = _act1_tokens()[:200]
    once = ps.score_noun_share(toks)
    twice = ps.score_noun_share(toks + toks)
    assert abs(twice - once) < 0.05, (once, twice)
    # ...and confirm score_nouns really does pay for the same trick, so this
    # test fails loudly if the two scorers ever get wired to the same thing.
    assert ps.score_nouns(toks + toks) > 1.8 * ps.score_nouns(toks)


def test_selective_beats_indiscriminate_at_equal_volume():
    toks = _act1_tokens()
    tags = ps.tag_sequence(toks)
    nouns = [t for t, g in zip(toks, tags) if g == ps.NOUN][:150]
    mixed = toks[:150]
    assert ps.score_noun_share(nouns) > 2 * ps.score_noun_share(mixed)


def test_floor_blocks_the_eat_one_noun_exploit():
    """Without the floored denominator, eating a single noun and stopping is
    a perfect score. With it, that is 1/30."""
    assert ps.score_noun_share(["king"]) <= 1.0 / ps.NOUN_SHARE_FLOOR + 1e-9


def test_bounded_and_empty_safe():
    assert ps.score_noun_share([]) == 0.0
    toks = _act1_tokens()
    for n in (10, 100, 500, 2000):
        s = ps.score_noun_share(toks[:n])
        assert 0.0 <= s <= 1.0, (n, s)


def test_indiscriminate_eating_lands_near_the_corpus_base_rate():
    """A worm with no selectivity should score roughly the base rate, whatever
    its volume — that is what makes the number readable without a denominator
    caveat, unlike score_nouns."""
    import random
    toks = _act1_tokens()
    rng = random.Random(0)
    for n in (100, 300, 600):
        s = ps.score_noun_share(rng.sample(toks, n))
        assert 0.15 < s < 0.40, (n, s)


def test_registered_in_scorers():
    assert ps.SCORERS["noun_share"] is ps.score_noun_share


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
