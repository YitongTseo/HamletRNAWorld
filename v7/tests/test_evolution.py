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


def test_repetitive_window_scores_less_than_diverse():
    diverse = [_win(["the", "king", "is", "dead", "and", "gone", "now",
                     "sleep", "crown", "grief", "cold", "night", "star",
                     "bell", "rose"], 90, 90)]
    repeat = [_win(["god"] * 15, 90, 90)]           # same judge E/C
    assert fitness(repeat) < fitness(diverse)
    # and the repeat is near the floor fraction of the diverse score
    assert fitness(repeat) <= ev.REP_MIN_FACTOR * fitness(diverse) + 1e-9


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
