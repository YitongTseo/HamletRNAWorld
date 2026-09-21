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
    """Fitness is exactly mean(1.5*(E/100)^γ + (C/100)^γ) with the divisor
    floored at FITNESS_WINDOW_FLOOR — no other factors. Regime change
    2026-08-15: was a plain sum, which paid for volume."""
    w = [_win(["a", "b", "c"], 80, 60)]
    per_win = ev.EMOTIONAL_WEIGHT * (0.8 ** ev.GAMMA) + (0.6 ** ev.GAMMA)
    assert abs(fitness(w) - per_win / ev.FITNESS_WINDOW_FLOOR) < 1e-12
    # at/above the floor it is a true mean: n identical windows == one window's
    # per-window value, independent of n
    full = [_win(["a", "b", "c"], 80, 60)] * ev.FITNESS_WINDOW_FLOOR
    assert abs(fitness(full) - per_win) < 1e-12
    assert abs(fitness(full * 2) - per_win) < 1e-12


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


# --- shape of a mutation (MUTATION_DF) --------------------------------------
#
# Children are drawn from a Student-t, not a Gaussian, because real mutational
# effect distributions are L-shaped and heavy-tailed rather than bell-shaped.
# The estimator stays exact because the update uses that distribution's SCORE
# (evolution.mutation_score), which for the t redescends — the heavy tail buys
# rare large mutations without letting one freak child drag the parent.

INF = float("inf")


def test_heavy_tail_keeps_the_variance_but_not_the_shape():
    """Same sigma must still mean the same sampling radius — the sigma
    controllers and the trust region both assume it — while genuinely large
    mutations become possible. Measured per coordinate, across children."""
    rng = np.random.default_rng(4)
    g = np.concatenate([ev.sample_eps((200,), rng, df=INF) for _ in range(1000)])
    t = np.concatenate([ev.sample_eps((200,), rng) for _ in range(1000)])
    assert abs(g.std() - 1.0) < 0.05 and abs(t.std() - 1.0) < 0.15
    assert np.mean(np.abs(g) > 5) < 1e-4
    assert np.mean(np.abs(t) > 5) > 1e-3
    assert np.abs(t).max() > 3 * np.abs(g).max()


def test_mutation_size_varies_between_children_not_just_within_them():
    """The property that per-coordinate tails DON'T give you, and the reason
    the sampler draws one chi-square scale per child rather than one per
    weight. Concentration of measure: sum 3,696 independent heavy-tailed
    coordinates and every child comes out the same size, which is the
    middling-change-everywhere behaviour the Gaussian was rejected for.

    Measured at d=3696, per-child ||eps||/sqrt(d): Gaussian spans 0.98-1.04
    from p05 to max; independent-t 0.91-2.90; shared-scale t 0.35-29.5."""
    rng = np.random.default_rng(6)
    d = 512
    def sizes(**kw):
        return np.array([np.linalg.norm(ev.sample_eps((d,), rng, **kw)) / np.sqrt(d)
                         for _ in range(2000)])
    g, t = sizes(df=INF), sizes()
    # A Gaussian gives every child the same-sized mutation at this dimension.
    assert np.percentile(g, 99) / np.median(g) < 1.15
    # The t gives most children a quiet genome and a few a large-effect one.
    assert np.percentile(t, 99) / np.median(t) > 2.0
    assert np.mean(t > 1.5 * np.median(t)) > 0.05
    assert np.mean(t < 0.7 * np.median(t)) > 0.05


def test_mutation_score_redescends_so_a_freak_child_cannot_hijack_the_step():
    """The property that makes the heavy tail safe. Under the Gaussian the
    influence of a child grows without bound in |eps|; under the t it peaks
    and falls away."""
    s = lambda e: float(ev.mutation_score(np.array([e]))[0])
    assert s(1.0) > s(0.25)                    # rises
    assert s(10.0) < s(1.0) / 4                # then redescends, hard
    assert s(30.0) < s(10.0)
    # The Gaussian is the identity — which is why the update could always be
    # written in terms of eps directly.
    for e in (0.5, 2.0, 40.0):
        assert ev.mutation_score(np.array([e]), df=INF)[0] == e


def test_gaussian_setting_reproduces_the_old_behaviour_exactly():
    """The A/B. df=inf must be the pre-2026-08-16 rule, bit for bit, in both
    the sampling and the update — otherwise the comparison is folklore.

    (df=None means 'use the module default', NOT Gaussian. Conflating the two
    silently turned this test into two identical runs when it was written.)"""
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    eps = ev.sample_eps((64,), rng_a, df=INF)
    assert np.array_equal(eps, rng_b.standard_normal(64))
    assert not np.array_equal(ev.sample_eps((64,), np.random.default_rng(11)),
                              np.random.default_rng(11).standard_normal(64))

    parent = np.zeros(6)
    eps_list = [np.full(6, 0.5), np.full(6, -0.5), np.full(6, 3.0)]
    scores = [1.0, 0.0, 2.0]
    saved = ev.MUTATION_DF
    try:
        ev.MUTATION_DF = INF
        gauss = ev.nes_update(parent, eps_list, scores, sigma=0.1)
        ev.MUTATION_DF = 3.0
        heavy = ev.nes_update(parent, eps_list, scores, sigma=0.1)
    finally:
        ev.MUTATION_DF = saved
    # The top-ranked child here is the 3-sigma outlier, so the Gaussian update
    # leans on it far harder than the heavy-tailed one does.
    assert np.linalg.norm(gauss) > 2 * np.linalg.norm(heavy)


def test_heavy_tailed_search_still_ascends():
    """Sanity: the estimator has to actually optimise. Hill-climb a quadratic
    with the real spawn/update pair and check the parent approaches the peak."""
    rng = np.random.default_rng(7)
    target = np.array([2.0, -1.0, 0.5, 3.0])
    parent = np.zeros(4)
    score = lambda x: -float(np.sum((x - target) ** 2))
    start = score(parent)
    sigma = 0.5
    for _ in range(60):
        children, eps = ev.spawn_population(parent, 12, sigma, rng)
        parent = ev.nes_update(parent, eps, [score(c) for c in children], sigma=sigma)
    assert score(parent) > start
    assert np.linalg.norm(parent - target) < np.linalg.norm(target) * 0.5
