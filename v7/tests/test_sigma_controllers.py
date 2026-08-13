"""Correctness tests for the Experiment-2 σ-control schemes.

These pin the MECHANISM, not the outcome (outcome needs a real landscape / the
live A/B): each scheme must move σ the right direction for a given success
signal, and the whole point of the redesign — that vs_centroid is σ-responsive
where vs_mean/vs_max are not — is exercised on a synthetic concave landscape.
"""
import math
import sys

import numpy as np

from server.evolution import SIGMA_GROW, SIGMA_SHRINK, SIGMA_MAX, SIGMA_MIN, SUCCESS_TARGET
from server.sigma_controllers import (
    GenContext, success_rate, one_fifth_step, xnes_step, ONE_FIFTH_SCHEMES,
)


def approx(a, b, tol=1e-9):
    assert math.isclose(a, b, rel_tol=1e-6, abs_tol=tol), f"{a} != {b}"


def test_success_rate_basic():
    assert success_rate([1, 2, 3, 4], 0.0) == 1.0        # all beat
    assert success_rate([1, 2, 3, 4], 100.0) == 0.0      # none beat
    assert success_rate([1, 2, 3, 4], 2.5) == 0.5        # half beat
    assert success_rate([1, 2, 3, 4], None) == SUCCESS_TARGET  # neutral hold
    assert success_rate([], 1.0) == SUCCESS_TARGET


def test_one_fifth_direction():
    scores = [10.0, 20.0, 30.0, 40.0]
    for scheme in ONE_FIFTH_SCHEMES:
        kw = {"vs_max": "prev_max", "vs_mean": "prev_fresh_mean",
              "vs_centroid": "centroid_fitness", "vs_elite": "elite_fitness"}[scheme]
        # baseline below everything -> success 1.0 -> grow
        hi, _ = one_fifth_step(scheme, 1.0, GenContext(scores, **{kw: 0.0}))
        approx(hi, SIGMA_GROW)
        # baseline above everything -> success 0.0 -> shrink
        lo, _ = one_fifth_step(scheme, 1.0, GenContext(scores, **{kw: 999.0}))
        approx(lo, SIGMA_SHRINK)


def test_sigma_clipped():
    scores = [1, 2, 3, 4]
    grown, _ = one_fifth_step("vs_mean", SIGMA_MAX, GenContext(scores, prev_fresh_mean=0.0))
    assert grown <= SIGMA_MAX
    shrunk, _ = one_fifth_step("vs_mean", SIGMA_MIN, GenContext(scores, prev_fresh_mean=999.0))
    assert shrunk >= SIGMA_MIN


def _concave_fitness(theta, x):
    # smooth concave bump peaked at theta; children farther out score lower
    return -float(np.sum((x - theta) ** 2))


def test_vs_mean_is_deaf_but_vs_centroid_responds_to_sigma():
    """The redesign's crux: on a concave landscape, the fraction of children
    beating the CHILD-MEAN stays ~0.5 regardless of σ (deaf → σ rails), while the
    fraction beating the CENTROID's own fitness FALLS as σ grows (responsive →
    has an interior fixed point)."""
    rng = np.random.default_rng(0)
    d = 20
    theta = np.zeros(d)
    f_centroid = _concave_fitness(theta, theta)  # = 0, the max
    for sigma in (0.1, 1.0, 3.0):
        srs_mean, srs_cent = [], []
        for _ in range(40):
            eps = rng.standard_normal((16, d))
            scores = [_concave_fitness(theta, theta + sigma * e) for e in eps]
            srs_mean.append(success_rate(scores, float(np.mean(scores))))
            srs_cent.append(success_rate(scores, f_centroid))
        mean_sr_mean = np.mean(srs_mean)
        mean_sr_cent = np.mean(srs_cent)
        # child-mean baseline sits ~0.5 for every sigma (deaf)
        assert 0.4 < mean_sr_mean < 0.6
        # centroid baseline: essentially nothing beats the true max, and it only
        # gets harder as sigma grows (monotone, σ-responsive)
        if sigma >= 1.0:
            assert mean_sr_cent < 0.15


def test_xnes_grows_when_winners_are_far_out():
    """If the best-scoring children are the ones sampled FARther from θ, xNES
    should push σ up; if they're the near-in ones, σ down."""
    d = 10
    eps = [np.ones(d) * 0.2, np.ones(d) * 0.5, np.ones(d) * 2.0, np.ones(d) * 3.0]
    far_best = [1.0, 2.0, 3.0, 4.0]   # score increases with distance -> grow
    near_best = [4.0, 3.0, 2.0, 1.0]  # score decreases with distance -> shrink
    assert xnes_step(1.0, eps, far_best) > 1.0
    assert xnes_step(1.0, eps, near_best) < 1.0


def test_sigma_anneal():
    from server.sigma_controllers import sigma_anneal_step
    # decays geometrically toward the floor, clipped to [MIN, MAX]
    assert sigma_anneal_step(3.0, decay=0.8, floor=0.1) == 3.0 * 0.8
    assert sigma_anneal_step(0.11, decay=0.8, floor=0.1) == 0.1        # floored
    assert SIGMA_MIN <= sigma_anneal_step(0.05) <= SIGMA_MAX


def test_compute_new_sigma_dispatch():
    from server.sigma_controllers import compute_new_sigma, GenContext
    ctx = GenContext([10, 20, 30, 40], prev_fresh_mean=5.0, elite_fitness=100.0)
    # vs_mean: everything beats mean=5 -> grow, baseline reported
    s, sr, base = compute_new_sigma("vs_mean", 1.0, ctx)
    assert sr == 1.0 and base == 5.0
    approx(s, SIGMA_GROW)
    # vs_elite: nothing beats elite=100 -> shrink
    s, sr, base = compute_new_sigma("vs_elite", 1.0, ctx)
    assert sr == 0.0 and base == 100.0
    approx(s, SIGMA_SHRINK)
    # xnes / anneal: no 1/5 signal -> sr and baseline are None
    s, sr, base = compute_new_sigma("sigma_anneal", 1.0, ctx)
    assert sr is None and base is None


def test_evolve_generation_all_schemes():
    """Integration: every scheme runs, keeps the population size, and keeps σ in
    range. vs_mean must still match the legacy baseline=prev_fresh_mean rule."""
    from server.evolution import evolve_generation, adapt_sigma
    N, d, n_elites = 16, 30, 2
    rng = np.random.default_rng(0)
    parent = np.zeros(d)
    sigma = 1.5
    epses = [rng.standard_normal(d) for _ in range(N)]
    genomes = [parent + sigma * e for e in epses]
    is_elite = [False] * N
    is_elite[0] = is_elite[1] = True          # two carried elites
    epses[0] = epses[1] = None                # elites aren't Gaussian samples
    fitnesses = list(rng.random(N) * 30)
    for scheme in ("vs_mean", "vs_elite", "xnes", "sigma_anneal"):
        ng = evolve_generation(parent, sigma, genomes, epses, is_elite, fitnesses,
                               n_elites=n_elites, rng=np.random.default_rng(1),
                               parent_fitness=15.0, scheme=scheme)
        assert len(ng.next_genomes) == N
        assert len(ng.next_epses) == N and len(ng.next_is_elite) == N
        assert SIGMA_MIN <= ng.new_sigma <= SIGMA_MAX
        assert ng.scheme == scheme
        if scheme == "vs_mean":  # backward-compat with the pre-Exp2 rule
            fresh = [fitnesses[i] for i in range(N) if not is_elite[i]]
            legacy_sr = sum(s > 15.0 for s in fresh) / len(fresh)
            approx(ng.new_sigma, adapt_sigma(sigma, legacy_sr))


if __name__ == "__main__":
    import traceback
    tests = [(n, f) for n, f in list(globals().items())
             if n.startswith("test_") and callable(f)]
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
