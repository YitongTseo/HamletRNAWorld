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
    rank_weights, nes_update, spawn_population, adapt_sigma,
    evolve_generation,
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


def test_unflatten_preserves_fractional_weights():
    """Change 1: the genome evolves in CONTINUOUS space. unflatten must NOT
    round to int — a sub-unit mutation like +0.3 has to survive write-back,
    otherwise the NES gradient (≈0.05/weight/gen) never crosses the rounding
    boundary and the genome stays frozen. The sim consumes weights as float
    (connectome.py: psyn[post] += w*scale) so fractional weights are valid."""
    keys = [("ASEL", "AVAL"), ("ASEL", "AVAR")]
    vec = np.array([5.3, -2.7])
    rt = unflatten_weights(vec, keys)
    assert rt["ASEL"]["AVAL"] == 5.3
    assert rt["ASEL"]["AVAR"] == -2.7


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


def test_fitness_consistency_beats_one_lucky_window():
    """Change 3: γ is lowered so the score is NOT dominated by a single
    lucky window. Ten coherent-ish windows (E/C=30) should now outweigh one
    spike (E/C=99) — otherwise selection rewards the one-off lottery the
    gardener flagged ("the lead passes to whoever got lucky") instead of a
    worm that is consistently more language-like. At the old γ=2.5 the spike
    won; at the lowered γ the body of decent windows wins."""
    one_top = [ScoredWindow(0, ["x"], 99, 99)]
    many_low = [ScoredWindow(i, ["x"], 30, 30) for i in range(10)]
    assert fitness(many_low) > fitness(one_top)


def test_fitness_higher_scores_still_win_per_window():
    """Lowering γ must not invert the basic ordering: a better window still
    scores higher than a worse one of the same count."""
    better = fitness([ScoredWindow(0, ["x"], 90, 90)])
    worse = fitness([ScoredWindow(0, ["x"], 30, 30)])
    assert better > worse


def test_fitness_volume_beyond_floor_does_not_pay():
    """Regime change 2026-08-15: fitness is a per-window MEAN (floored
    divisor), not a sum. Doubling the window count at slightly lower quality
    must now LOSE — under the old sum, 24 windows of 55 beat 12 of 60, which
    let big eaters out-rank better poets (data-lifelike-2 gen 6)."""
    quality = [ScoredWindow(i, ["x"], 60, 60) for i in range(12)]
    volume = [ScoredWindow(i, ["x"], 55, 55) for i in range(24)]
    assert fitness(quality) > fitness(volume)


def test_fitness_floor_blocks_lucky_short_life():
    """The divisor floor: a worm that died after 3 lucky windows must not
    out-rank a full life of decent ones. An unfloored mean would score the
    short life ~95 vs ~80; the floor dilutes it to 3·q95/12."""
    short_lucky = [ScoredWindow(i, ["x"], 95, 95) for i in range(3)]
    full_decent = [ScoredWindow(i, ["x"], 80, 80) for i in range(13)]
    assert fitness(full_decent) > fitness(short_lucky)


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


def test_adapt_sigma_rechenberg_1_5_rule():
    """Change 4: σ adapts off the SUCCESS RATE (fraction of fresh children
    that beat the incumbent), not a single noisy best-score flag. Classic
    Rechenberg 1/5 rule: success > 1/5 means steps are too timid → grow;
    success < 1/5 means we're mostly missing → shrink; ~1/5 holds steady.
    This is far more stable than reacting to one lucky window."""
    assert adapt_sigma(1.0, success_rate=0.5) > 1.0     # succeeding a lot → grow
    assert adapt_sigma(1.0, success_rate=0.0) < 1.0     # mostly failing → shrink
    assert adapt_sigma(1.0, success_rate=0.2) == 1.0    # right at 1/5 → hold


def test_adapt_sigma_clips_to_range():
    """Repeated grow / shrink must not run sigma to zero or infinity."""
    s = 1.0
    for _ in range(100):
        s = adapt_sigma(s, success_rate=1.0)
    assert s <= SIGMA_MAX
    s = 1.0
    for _ in range(100):
        s = adapt_sigma(s, success_rate=0.0)
    assert s >= SIGMA_MIN


def test_spawn_population_all_perturbed():
    """Change 5: spawn_population produces N *fresh* samples (no forced
    elite-copy — elitism is handled one level up by carrying real genomes).
    Each child must equal parent + sigma*eps for its recorded eps."""
    rng = np.random.default_rng(0)
    parent = np.array([1.0, 2.0, 3.0])
    children, eps = spawn_population(parent, n=4, sigma=0.5, rng=rng)
    assert len(children) == 4 and len(eps) == 4
    for c, e in zip(children, eps):
        assert not np.array_equal(c, parent)
        assert np.allclose(c, parent + 0.5 * e)


# --- evolve_generation: the elitist-NES step ------------------------------
# evolve_generation(parent, sigma, genomes, epses, is_elite, fitnesses,
#                   n_elites, rng, prev_best) -> NextGen
# It (a) updates the parent via NES using only fresh children's TRUE eps,
# (b) carries the top-n_elites genomes forward verbatim, (c) fills the rest
# with fresh samples of the updated parent, (d) adapts sigma off success rate.

def _evolve(parent, sigma, genomes, epses, is_elite, fitnesses,
            n_elites=1, seed=0, prev_best=None):
    return evolve_generation(
        parent_vec=np.asarray(parent, dtype=float),
        sigma=sigma,
        genomes=[np.asarray(g, dtype=float) for g in genomes],
        epses=[None if e is None else np.asarray(e, dtype=float) for e in epses],
        is_elite=is_elite,
        fitnesses=fitnesses,
        n_elites=n_elites,
        rng=np.random.default_rng(seed),
        prev_best_fitness=prev_best,
    )


def test_evolve_carries_top_elites_verbatim():
    """The top-n_elites genomes (by fitness) must reappear UNCHANGED in the
    next generation — this is the 'hold what's good' the gardener begged for.
    Here worm #2 has the best fitness, worm #0 second; with n_elites=2 both
    their exact genomes must be present in next_genomes."""
    parent = [0.0, 0.0]
    genomes = [[1.0, 1.0], [2.0, 2.0], [9.0, 9.0], [3.0, 3.0]]
    epses = [[1, 1], [2, 2], [9, 9], [3, 3]]  # fresh (non-elite) this gen
    res = _evolve(parent, 0.5, genomes, epses,
                  is_elite=[False] * 4, fitnesses=[5.0, 1.0, 99.0, 4.0],
                  n_elites=2, seed=1)
    survivors = [g.tolist() for g in res.next_genomes]
    assert [9.0, 9.0] in survivors      # best
    assert [1.0, 1.0] in survivors      # second best (worm #0)


def test_evolve_fresh_children_are_parent_plus_sigma_eps():
    """Non-elite slots in the next gen must be genuine samples of the UPDATED
    parent: genome == new_parent + new_sigma * its recorded eps."""
    parent = [0.0, 0.0]
    genomes = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    res = _evolve(parent, 0.5, genomes, epses=[[1, 0], [0, 1], [1, 1]],
                  is_elite=[False] * 3, fitnesses=[3.0, 2.0, 1.0],
                  n_elites=1, seed=2)
    for genome, eps, elite in zip(res.next_genomes, res.next_epses, res.next_is_elite):
        if not elite:
            assert eps is not None
            assert np.allclose(genome, res.new_parent + res.new_sigma * eps)


def test_evolve_gradient_ignores_elite_genomes():
    """Change 2/5: an elite carried from a prior gen is NOT a Gaussian sample
    of the current parent, so it must be EXCLUDED from the NES gradient. Here
    the elite has a huge fitness but eps=None; the parent must move toward the
    high-scoring FRESH child (eps=[1,0]), not get yanked by the elite."""
    parent = [0.0, 0.0]
    genomes = [[5.0, 5.0], [1.0, 0.0], [0.0, 1.0]]
    epses = [None, [1.0, 0.0], [0.0, 1.0]]
    res = _evolve(parent, 0.5, genomes, epses,
                  is_elite=[True, False, False], fitnesses=[100.0, 10.0, 1.0],
                  n_elites=1, seed=3)
    # fresh child [1,0] outscores [0,1] → parent moves +x more than +y
    assert res.new_parent[0] > res.new_parent[1]
    # and the elite genome still survives verbatim
    assert [5.0, 5.0] in [g.tolist() for g in res.next_genomes]


def test_evolve_success_rate_drives_sigma():
    """σ should grow when most fresh children beat the incumbent (steps too
    small) and shrink when none do (steps too large)."""
    parent = [0.0, 0.0]
    genomes = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    epses = [[1, 0], [0, 1], [1, 1]]
    grew = _evolve(parent, 1.0, genomes, epses, is_elite=[False] * 3,
                   fitnesses=[10.0, 10.0, 10.0], n_elites=1, seed=4, prev_best=1.0)
    shrank = _evolve(parent, 1.0, genomes, epses, is_elite=[False] * 3,
                     fitnesses=[0.1, 0.1, 0.1], n_elites=1, seed=4, prev_best=1.0)
    assert grew.new_sigma > 1.0
    assert shrank.new_sigma < 1.0


def test_evolve_next_generation_is_full_size():
    """next_genomes must have one entry per worm slot (n_elites + fresh)."""
    parent = [0.0, 0.0]
    genomes = [[float(i), 0.0] for i in range(6)]
    res = _evolve(parent, 0.5, genomes, epses=[[1, 0]] * 6,
                  is_elite=[False] * 6, fitnesses=[float(i) for i in range(6)],
                  n_elites=2, seed=5)
    assert len(res.next_genomes) == 6
    assert len(res.next_epses) == 6
    assert len(res.next_is_elite) == 6
    assert sum(res.next_is_elite) == 2  # exactly two elites carried


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
