"""Tests for the POS-tagger fitness functions, focused on the revamped
`score_nouns` reward (nouns rewarded, other words penalized)."""
from __future__ import annotations

from server import pos_scorers as ps


def test_score_nouns_empty():
    assert ps.score_nouns([]) == 0.0


def test_score_nouns_rewards_nouns_penalizes_others():
    # Tag the words directly so the test asserts on the *formula*, not on
    # NLTK's tagging of any particular sentence (which can drift by version).
    tags = {
        "king": ps.NOUN, "ghost": ps.NOUN,        # nouns: +1 each
        "quickly": ps.ADV, "the": ps.DET,          # others: -0.2 each
        ".": ".",                                   # punctuation: ignored
    }
    eaten = ["king", "ghost", "quickly", "the", "."]
    expected = 2 * 1.0 + 2 * (-ps.NOUN_OTHER_PENALTY)  # 2 - 0.4 = 1.6

    orig = ps.tag_sequence
    ps.tag_sequence = lambda words: [tags[w] for w in words]
    try:
        assert ps.score_nouns(eaten) == expected
    finally:
        ps.tag_sequence = orig


def test_score_nouns_selectivity_beats_volume():
    # A selective worm (pure nouns) must beat a high-volume worm that eats
    # the same nouns plus a pile of junk — the whole point of the penalty.
    nouns = ["king"] * 200
    junk = ["the"] * 500

    orig = ps.tag_sequence
    ps.tag_sequence = lambda words: [
        ps.NOUN if w == "king" else ps.DET for w in words
    ]
    try:
        selective = ps.score_nouns(nouns)
        high_volume = ps.score_nouns(nouns + junk)
        assert selective == 200.0
        assert high_volume < selective
        # Pure junk goes negative.
        assert ps.score_nouns(junk) < 0
    finally:
        ps.tag_sequence = orig


def test_penalty_is_between_zero_and_one():
    # A single noun must always outweigh a single non-noun's cost, so the
    # worm is never better off eating nothing than eating one good word.
    assert 0.0 < ps.NOUN_OTHER_PENALTY < 1.0
