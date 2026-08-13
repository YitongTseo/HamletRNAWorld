"""Tests for v7 evolution changes: repetition penalty + σ-collapse fix."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import evolution as ev
from server.evolution import repetition_factor, fitness, evolve_generation
from server.judge import ScoredWindow


def _win(tokens, e, c):
    return ScoredWindow(idx=0, tokens=tokens, emotional=e, coherence=c)


def test_repetition_factor_bounds():
    assert repetition_factor([]) == 1.0
    assert repetition_factor(["a"]) == 1.0
    # long fully-repeated window -> floor
    assert repetition_factor(["god"] * 15) == ev.REP_MIN_FACTOR
    # all distinct -> 1.0
    assert repetition_factor([f"w{i}" for i in range(15)]) == 1.0
    # natural-ish diversity (ratio >= floor) -> unpenalized
    assert repetition_factor(["the", "king", "the", "queen", "and", "the",
                              "lord", "of", "the", "north"]) == 1.0
    # case-insensitive: casing doesn't create diversity
    assert repetition_factor(["God", "god", "GOD"]) == repetition_factor(["a", "a", "a"])
    assert repetition_factor(["God", "god", "GOD"]) < 1.0


def test_repetition_is_not_penalized_in_fitness():
    """2026-08-13: let them repeat. `fitness` no longer applies the token
    diversity discount, so two windows the judge rated identically score
    identically regardless of how repetitive they are."""
    diverse = [_win(["the", "king", "is", "dead", "and", "gone", "now",
                     "sleep", "crown", "grief", "cold", "night", "star",
                     "bell", "rose"], 90, 90)]
    repeat = [_win(["god"] * 15, 90, 90)]           # same judge E/C
    assert fitness(repeat) == fitness(diverse)


def test_fitness_is_pure_judge_score():
    """Fitness is exactly Σ 1.5*(E/100)^γ + (C/100)^γ — no other factors."""
    w = [_win(["a", "b", "c"], 80, 60)]
    expected = ev.EMOTIONAL_WEIGHT * (0.8 ** ev.GAMMA) + (0.6 ** ev.GAMMA)
    assert abs(fitness(w) - expected) < 1e-12


# --- step size vs sigma (the decoupling fix) ------------------------------

def _step_norm(sigma, d=3689, n=11, seed=0, **kw):
    rng = np.random.default_rng(seed)
    parent = np.zeros(d)
    eps = [rng.standard_normal(d) for _ in range(n)]
    scores = list(range(n))[::-1]
    return float(np.linalg.norm(
        ev.nes_update(parent, eps, scores, sigma=sigma, **kw) - parent))


def test_step_scales_linearly_with_sigma():
    """THE regression this fixes. The step used to carry a 1/sigma factor, so
    shrinking sigma GREW the step; it must now scale WITH sigma."""
    a, b = _step_norm(0.02), _step_norm(0.04)
    assert b > a
    assert abs(b / a - 2.0) < 1e-6, f"step must double when sigma doubles: {b/a}"


def test_step_to_sampling_radius_ratio_is_sigma_invariant():
    """The property that makes any sigma controller viable: however sigma
    moves, the parent always steps the same FRACTION of the region sampled."""
    d = 3689
    ratios = [_step_norm(s, d=d) / (s * np.sqrt(d))
              for s in (0.02, 0.1, 0.5, 3.0)]
    assert max(ratios) - min(ratios) < 1e-9, ratios
    # and that fraction is a sane fraction of the sampled cloud, not 40x it
    assert 0.01 < ratios[0] < ev.TRUST_RADIUS


def test_old_formula_would_have_exploded_at_small_sigma():
    """Documents the bug: with the 1/sigma form the parent moved 40x further
    than it sampled once sigma hit the floor."""
    d, n = 3689, 11
    rng = np.random.default_rng(0)
    eps = [rng.standard_normal(d) for _ in range(n)]
    rw = ev.rank_weights(n)
    weighted = sum(rw[r] * eps[r] for r in range(n))
    sigma = 0.02
    old_step = np.linalg.norm((0.1 / (n * sigma)) * weighted)
    sampling_radius = sigma * np.sqrt(d)
    assert old_step / sampling_radius > 30          # the pathology
    assert _step_norm(sigma) / sampling_radius < 1  # the fix


def test_trust_region_caps_a_runaway_step():
    d, n = 500, 8
    parent = np.zeros(d)
    # one enormous eps -> huge rank-weighted sum
    eps = [np.full(d, 1000.0)] + [np.zeros(d) for _ in range(n - 1)]
    scores = list(range(n))[::-1]
    sigma = 0.1
    out = ev.nes_update(parent, eps, scores, sigma=sigma, lr=50.0)
    cap = ev.TRUST_RADIUS * sigma * np.sqrt(d)
    assert np.linalg.norm(out - parent) <= cap * (1 + 1e-9)


def test_trust_region_not_binding_in_normal_use():
    """The cap is a safety net; ordinary generations must sit under it so the
    gradient magnitude still carries information."""
    d = 3689
    for s in (0.02, 0.1, 1.0):
        assert _step_norm(s, d=d) < ev.TRUST_RADIUS * s * np.sqrt(d)


def test_zero_sigma_is_a_noop():
    d = 100
    parent = np.ones(d)
    eps = [np.random.default_rng(1).standard_normal(d) for _ in range(4)]
    out = ev.nes_update(parent, eps, [3, 2, 1, 0], sigma=0.0)
    assert np.allclose(out, parent)


def test_step_direction_still_follows_the_ranking():
    """Rescaling must not change WHICH way we move."""
    d, n = 200, 6
    rng = np.random.default_rng(4)
    parent = np.zeros(d)
    eps = [rng.standard_normal(d) for _ in range(n)]
    scores = [10.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    step = ev.nes_update(parent, eps, scores, sigma=0.3) - parent
    # positive projection onto the best child's noise, negative onto the worst
    assert float(step @ eps[0]) > 0
    assert float(step @ eps[-1]) < 0


def _run_gen(parent_fitness, fresh_scores, sigma=0.1):
    """Drive one evolve_generation with all-fresh children whose fitnesses we
    control, returning the NextGen."""
    n = len(fresh_scores)
    d = 8
    rng = np.random.default_rng(0)
    parent = np.zeros(d)
    genomes = [parent + sigma * rng.standard_normal(d) for _ in range(n)]
    epses = [(g - parent) / sigma for g in genomes]
    is_elite = [False] * n
    return evolve_generation(
        parent, sigma, genomes, epses, is_elite, fresh_scores,
        n_elites=0, rng=rng, parent_fitness=parent_fitness,
    )


def test_sigma_grows_when_most_children_beat_baseline():
    # 90% of children beat the parent baseline -> success >> 1/5 -> σ grows
    ng = _run_gen(parent_fitness=1.0, fresh_scores=[2.0] * 9 + [0.0], sigma=0.1)
    assert ng.success_rate > ev.SUCCESS_TARGET
    assert ng.new_sigma > 0.1


def test_sigma_shrinks_when_few_children_beat_baseline():
    # only 1/10 beats the baseline -> success < 1/5 -> σ shrinks
    ng = _run_gen(parent_fitness=1.0, fresh_scores=[2.0] + [0.0] * 9, sigma=0.1)
    assert ng.success_rate < ev.SUCCESS_TARGET
    assert ng.new_sigma < 0.1


def test_fresh_mean_reported():
    ng = _run_gen(parent_fitness=None, fresh_scores=[1.0, 3.0, 5.0], sigma=0.1)
    assert abs(ng.fresh_mean - 3.0) < 1e-9


def test_champion_baseline_would_have_collapsed():
    """Regression guard: under v6's 'beat the max' rule, a stationary noisy
    population almost never improves on its own best, so σ only ever shrinks.
    The population-baseline rule lets a stationary population hold/grow σ."""
    scores = [1.0, 1.2, 0.8, 1.1, 0.9, 1.05, 0.95, 1.15, 0.85, 1.0]
    champ = max(scores)                     # v6 baseline
    mean = float(np.mean(scores))           # v7 baseline (proxy)
    beat_champ = sum(s > champ for s in scores) / len(scores)
    beat_mean = sum(s > mean for s in scores) / len(scores)
    assert beat_champ == 0.0                # nobody beats the champion -> only shrink
    assert beat_mean >= 0.4                 # ~half beat the mean -> σ can breathe


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
