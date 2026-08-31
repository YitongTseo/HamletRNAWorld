"""WORMLET_EVOLVE_MASK — freeze the motor circuitry, evolve only the nose.

The arithmetic that motivates this: an ES gradient from `lambda` children in
`d` dimensions aligns with the true gradient by roughly sqrt(lambda/d). On this
connectome with 5 fresh children that is sqrt(5/3689) = 0.037, so every step is
96% random. Restricting to the 228 synapses leaving the 24 amphid neurons takes
it to 0.148 — 4x better, free — where tripling the population buys 1.7x at
triple the cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import evolution as E


def _keys_and_parent():
    import json
    import server.orchestrator as orch
    w = json.loads(orch.DEFAULT_WEIGHTS.read_text())
    vec, keys = E.flatten_weights(w)
    return vec, keys


def test_all_returns_none_so_the_unmasked_path_is_untouched():
    _vec, keys = _keys_and_parent()
    assert E.build_evolve_mask(keys, "all") is None
    assert E.build_evolve_mask(keys, "") is None


def test_chemo_mask_is_a_small_slice_of_the_genome():
    _vec, keys = _keys_and_parent()
    m = E.build_evolve_mask(keys, "chemo")
    m2 = E.build_evolve_mask(keys, "chemo2")
    assert 0 < m.sum() < m2.sum() < len(keys)
    # chemo must be a strict subset of chemo2
    assert bool((m & ~m2).sum()) is False
    # and it must be a big reduction, or it is not worth the complexity
    assert m.sum() < 0.15 * len(keys)


def test_lifelike_genes_are_always_evolvable():
    """They are the learning machinery, not locomotion — freezing them would
    switch off the search over how the worm learns."""
    from sim.lifelike import LIFELIKE_KEY
    keys = [("AWCL", "AIYL"), (LIFELIKE_KEY, "eta"), ("VB01", "DB02")]
    m = E.build_evolve_mask(keys, "chemo")
    assert m[1], "the _lifelike block must stay in the search space"
    assert not m[2], "motor synapses must be frozen"


def test_unknown_mode_is_a_loud_error():
    _vec, keys = _keys_and_parent()
    for bad in ("motor", "chemo3", "228"):
        try:
            E.build_evolve_mask(keys, bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should have raised")


def _run(mask, seed=0, n=8, d=40):
    rng = np.random.default_rng(seed)
    parent = np.arange(d, dtype=float)
    genomes = [parent + 0.01 * i for i in range(n)]
    epses = [rng.standard_normal(d) for _ in range(n)]
    fits = [float(i) for i in range(n)]
    return E.evolve_generation(
        parent, 0.5, genomes, epses, [False] * n, fits,
        n_elites=3, rng=np.random.default_rng(seed), mask=mask)


def test_frozen_coordinates_never_move():
    d = 40
    mask = np.zeros(d, dtype=bool)
    mask[:10] = True
    ng = _run(mask)
    parent = np.arange(d, dtype=float)
    assert np.allclose(ng.new_parent[~mask], parent[~mask]), \
        "the parent moved on a frozen coordinate"
    for child, is_elite in zip(ng.next_genomes, ng.next_is_elite):
        if not is_elite:
            assert np.allclose(child[~mask], parent[~mask]), \
                "a fresh child was mutated on a frozen coordinate"


def test_eps_is_returned_full_length_and_zero_where_frozen():
    """Full-length eps keeps the on-disk format unchanged, so switching the
    mask mid-lineage can never cause a length mismatch."""
    d = 40
    mask = np.zeros(d, dtype=bool)
    mask[:10] = True
    ng = _run(mask)
    for e, is_elite in zip(ng.next_epses, ng.next_is_elite):
        if is_elite:
            assert e is None
        else:
            assert e.shape == (d,)
            assert np.allclose(e[~mask], 0.0)
            assert np.any(e[mask] != 0.0)


def test_masked_and_unmasked_agree_when_the_mask_is_everything():
    d = 40
    ng_none = _run(None)
    ng_all = _run(np.ones(d, dtype=bool))
    assert np.allclose(ng_none.new_parent, ng_all.new_parent)
    for a, b in zip(ng_none.next_genomes, ng_all.next_genomes):
        assert np.allclose(a, b)


def test_elites_keep_their_own_frozen_half():
    """Elites carry their OWN genome, not the parent's frozen coordinates —
    the guard that stops a mid-lineage mask change rewriting history."""
    d, n = 20, 6
    mask = np.zeros(d, dtype=bool)
    mask[:5] = True
    parent = np.zeros(d)
    genomes = [np.full(d, float(i)) for i in range(n)]   # distinct frozen halves
    rng = np.random.default_rng(1)
    ng = E.evolve_generation(
        parent, 0.5, genomes, [rng.standard_normal(d) for _ in range(n)],
        [False] * n, [float(i) for i in range(n)],
        n_elites=2, rng=np.random.default_rng(1), mask=mask)
    best = max(range(n), key=lambda i: float(i))
    assert np.allclose(ng.next_genomes[0][~mask], genomes[best][~mask])
