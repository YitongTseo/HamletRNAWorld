"""Tests for the generational evolution pieces — punct cleanup, NES math,
end-of-corpus detection, judge windowing/sampling. No Claude API calls."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corpus.hamlet import get_sentences
from server.poem_clean import clean
from server.evolution import (
    fitness, flatten_weights, unflatten_weights,
    rank_weights, nes_update, spawn_children, adapt_sigma,
    SIGMA_MIN, SIGMA_MAX,
)
from server.judge import make_windows, sample_windows, _parse_scores, ScoredWindow
from sim.text_scroller import TextScroller


# --- punctuation cleanup --------------------------------------------------

def test_punct_collapse_run_keep_first():
    """Spec example 1."""
    assert clean(["he", ".", ".", "."]) == ["he", "."]


def test_punct_collapse_mixed_run_keep_first():
    """Spec example 2 — different punctuation in the same run still collapses
    to the FIRST mark seen."""
    assert clean(["I", "'tis", "!", "?", ".", "."]) == ["I", "'tis", "!"]


def test_punct_preserve_trailing():
    """Spec example 3 — a trailing run collapses to one mark, NOT stripped."""
    got = clean(["to", "be", ",", "or", "not", "to", "be", ".", ".", ".", "."])
    assert got == ["to", "be", ",", "or", "not", "to", "be", "."]


def test_punct_empty_input():
    assert clean([]) == []


def test_punct_no_punctuation():
    assert clean(["just", "plain", "words"]) == ["just", "plain", "words"]


def test_punct_alternating_runs():
    """Multiple separate runs each collapse to their own first mark."""
    got = clean(["a", ".", ".", "b", ",", ",", ",", "c", "!", "?"])
    assert got == ["a", ".", "b", ",", "c", "!"]


# --- weight flatten / unflatten round-trip --------------------------------

def test_flatten_unflatten_roundtrip_with_negatives():
    """Connectome weights can be negative (inhibitory); the round-trip must
    preserve sign."""
    w = {
        "ASEL": {"AVAL": 5, "AVAR": -3},
        "ASER": {"AVAL": -7, "AVAR": 2},
    }
    vec, keys = flatten_weights(w)
    rt = unflatten_weights(vec, keys)
    assert rt == w


def test_flatten_is_sorted_deterministic():
    """Same dict must always flatten to the same key order regardless of
    insertion order — otherwise NES indices drift across runs."""
    a = {"B": {"y": 1, "x": 2}, "A": {"z": 3}}
    b = {"A": {"z": 3}, "B": {"x": 2, "y": 1}}
    _, keys_a = flatten_weights(a)
    _, keys_b = flatten_weights(b)
    assert keys_a == keys_b == [("A", "z"), ("B", "x"), ("B", "y")]


# --- fitness math ---------------------------------------------------------

def test_fitness_empty_is_zero():
    assert fitness([]) == 0.0


def test_fitness_top_window_dominates():
    """A single near-100 window must outweigh many low-30s windows because
    of γ=2.5 — that's the whole point of the non-linear shape.
    Quantitatively: (0.99)^2.5 ≈ 0.975, (0.3)^2.5 ≈ 0.049 — ~20× ratio per
    window, so 1 top beats 10 lows."""
    one_top = [ScoredWindow(0, ["x"], 99, 99)]
    many_low = [ScoredWindow(i, ["x"], 30, 30) for i in range(10)]
    assert fitness(one_top) > fitness(many_low)


def test_fitness_emotional_weighted_above_coherence():
    """Same total, but flipping E↑/C↓ vs E↓/C↑ shouldn't be symmetric:
    the 1.5× emotional weight prefers high E."""
    high_e = [ScoredWindow(0, ["x"], 80, 40)]
    high_c = [ScoredWindow(0, ["x"], 40, 80)]
    assert fitness(high_e) > fitness(high_c)


# --- NES math -------------------------------------------------------------

def test_rank_weights_sum_to_zero():
    """The NES update is a zero-mean weighted average; if rank_weights
    didn't sum to zero, the parent would drift even when fitness is flat."""
    for n in [2, 6, 10, 40]:
        assert abs(rank_weights(n).sum()) < 1e-9


def test_rank_weights_descending():
    """Best-ranked must get highest weight."""
    rw = rank_weights(10)
    for i in range(len(rw) - 1):
        assert rw[i] >= rw[i + 1], f"rank_weights({i}) < rank_weights({i+1})"


def test_nes_update_moves_toward_best_eps():
    """A single child with high score and a known eps must push the parent
    in that eps direction (this is the whole reason NES isn't a random walk).
    """
    parent = np.zeros(3)
    eps_list = [
        np.zeros(3),                    # elite (no perturbation)
        np.array([1.0, 0.0, 0.0]),      # child 1 — high score
        np.array([0.0, 1.0, 0.0]),      # child 2 — low score
    ]
    scores = [0.5, 10.0, 0.1]
    new_parent = nes_update(parent, eps_list, scores, sigma=0.5, lr=0.5)
    # Most of the movement should be along eps_1 (the high-scorer)
    assert new_parent[0] > new_parent[1], "should move toward eps_1 more than eps_2"
    assert new_parent[2] == 0.0, "no movement along an unexplored axis"


def test_nes_update_no_change_when_eps_all_zero():
    """Cold-start case: no children spawned yet, so eps_i = 0 ∀ i. Update
    must leave parent unchanged."""
    parent = np.array([1.0, 2.0, 3.0])
    eps_list = [np.zeros(3) for _ in range(6)]
    scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    new_parent = nes_update(parent, eps_list, scores, sigma=0.5)
    assert np.allclose(new_parent, parent)


def test_adapt_sigma_shrinks_on_improvement():
    """1/5-rule: shrink when we found a better local optimum, grow when stuck."""
    s = 1.0
    s = adapt_sigma(s, improved=True)
    assert s < 1.0
    s = adapt_sigma(s, improved=False)
    s = adapt_sigma(s, improved=False)
    assert s > SIGMA_MIN


def test_adapt_sigma_clips_to_range():
    """Repeated shrink / grow must not run sigma to zero or infinity."""
    s = 1.0
    for _ in range(100):
        s = adapt_sigma(s, improved=True)
    assert s >= SIGMA_MIN
    s = 1.0
    for _ in range(100):
        s = adapt_sigma(s, improved=False)
    assert s <= SIGMA_MAX


def test_spawn_children_elite_is_unchanged():
    """First child must be the elite — exact copy of the parent."""
    rng = np.random.default_rng(0)
    parent = np.array([1.0, 2.0, 3.0])
    children, eps = spawn_children(parent, n=4, sigma=0.5, rng=rng)
    assert np.array_equal(children[0], parent)
    assert np.array_equal(eps[0], np.zeros_like(parent))
    # The others should have actual perturbations
    for c, e in zip(children[1:], eps[1:]):
        assert not np.array_equal(c, parent)
        assert not np.array_equal(e, np.zeros_like(parent))


# --- end-of-corpus detection ----------------------------------------------

def test_corpus_exhausted_after_one_pass():
    """TextScroller in loop=False mode must report corpus_exhausted=True
    after every sentence has been spawned AND scrolled off the top."""
    sentences = get_sentences("opening")
    ts = TextScroller(sentences, loop=False)
    # Step in 0.5s chunks until exhausted or we hit a sanity limit.
    for _ in range(2000):
        ts.step(0.5)
        if ts.corpus_exhausted:
            break
    assert ts.corpus_exhausted, f"never exhausted after sent_idx={ts._sent_idx}, active={len(ts._active)}"
    assert ts._sent_idx == len(sentences)


def test_corpus_exhausted_false_during_loop_mode():
    """loop=True (legacy) never reports corpus_exhausted, even after the
    sentence list has been cycled through many times."""
    sentences = get_sentences("opening")
    ts = TextScroller(sentences, loop=True)
    for _ in range(5000):
        ts.step(0.5)
    assert not ts.corpus_exhausted


# --- judge windowing + sampling -------------------------------------------

def test_make_windows_non_overlapping():
    """stride=window_size should produce contiguous non-overlapping windows."""
    tokens = list("abcdefghij")  # 10 tokens
    w = make_windows(tokens, window_size=3, stride=3)
    assert w == [(0, ["a","b","c"]), (3, ["d","e","f"]),
                 (6, ["g","h","i"]), (9, ["j"])]


def test_sample_windows_deterministic_by_seed():
    """Same seed → same sampling; different seed → different sampling.
    Otherwise the audit trail of which windows got judged isn't
    reproducible across re-runs."""
    tokens = list(range(100))
    w = make_windows([str(t) for t in tokens], window_size=10, stride=10)
    a = sample_windows(w, fraction=0.3, seed=42)
    b = sample_windows(w, fraction=0.3, seed=42)
    c = sample_windows(w, fraction=0.3, seed=99)
    assert a == b
    assert a != c


def test_parse_scores_clamps_out_of_range():
    """LLM might emit out-of-range positive numbers; we clamp to 1-100
    instead of dropping the line. Negative numbers don't match the regex
    (only \\d+ allowed) — those lines are silently skipped, which is fine
    because we'd rather lose a window than score it incorrectly."""
    out = _parse_scores("0,150,99\n1,200,300\n2,0,0")
    assert out[0] == (100, 99)
    assert out[1] == (100, 100)
    assert out[2] == (1, 1)  # zero clamped up to 1


def test_parse_scores_ignores_noise():
    """Anything that doesn't match `\\d+,\\d+,\\d+` is skipped silently."""
    out = _parse_scores("preamble\n0,80,70\n  blah blah\n1,60,50\nend.")
    assert out == {0: (80, 70), 1: (60, 50)}


if __name__ == "__main__":
    # Manual runner: no pytest dependency, just print pass/fail per function.
    import traceback
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
