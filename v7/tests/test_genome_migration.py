"""Retrofitting the _lifelike genes into a lineage that predates them.

The eight poetry flasks were 137 generations deep when these genes were
added, so their parent_keys had no slot for them and evolution could never
tune the learning rules. These tests pin the three properties that make the
mid-flight migration safe, and the per-dimension NES scaling that makes it
useful.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import evolution, generations  # noqa: E402
from sim import lifelike  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class _FakeWorm:
    """Just enough Worm for migrate_genome_layout: a weights.json beside a
    poem_path."""

    def __init__(self, root: Path, name: str, genome: dict):
        self.name = name
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        self.poem_path = d / "poem.txt"
        self.poem_path.write_text("")
        (d / "weights.json").write_text(json.dumps(genome))

    def genome(self) -> dict:
        return json.loads((self.poem_path.parent / "weights.json").read_text())


def _stock_genome() -> dict:
    """A connectome-only genome, like a pre-lifelike lineage's."""
    return {"ADAL": {"ADAR": 3.0, "AIBL": -2.0},
            "VD9": {"VD10": 1.5, "PDER": 4.0}}


def _state_from(genome: dict, **kw):
    vec, keys = evolution.flatten_weights(genome)
    return generations.GenerationState(
        group_name="flask_test", generation=137,
        sigma=0.02,
        parent_vector=vec.tolist(),
        parent_keys=[list(k) for k in keys],
        **kw)


class _EnvOn:
    """Turn the lifelike flags on for the duration of a block."""

    KEYS = ("WORMLET_PLASTICITY", "WORMLET_HUNGER", "WORMLET_HABITUATION")

    def __enter__(self):
        self._old = {k: os.environ.get(k) for k in self.KEYS}
        for k in self.KEYS:
            os.environ[k] = "1"

    def __exit__(self, *a):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --------------------------------------------------------------------------
# the three safety properties
# --------------------------------------------------------------------------

def test_lifelike_block_sorts_last_so_migration_is_an_append():
    """The load-bearing fact. flatten_weights orders by source name and '_'
    (0x5F) outranks every uppercase neuron name, so the genes append rather
    than shifting existing indices. If neuron naming ever changed to include
    lowercase, this breaks and 137 generations of weights get reinterpreted.
    """
    genome = _stock_genome()
    before_vec, before_keys = evolution.flatten_weights(genome)
    lifelike.ensure_params(genome)
    after_vec, after_keys = evolution.flatten_weights(genome)

    n = len(before_keys)
    assert after_keys[:n] == before_keys, "existing genome indices moved"
    assert np.array_equal(after_vec[:n], before_vec), "existing weights changed"
    assert all(k[0] == lifelike.LIFELIKE_KEY for k in after_keys[n:])
    assert len(after_keys) == n + len(lifelike.PARAM_SPEC)


def test_migration_extends_state_and_every_worm_together():
    with _EnvOn(), tempfile.TemporaryDirectory() as td:
        root = Path(td)
        genome = _stock_genome()
        state = _state_from(genome)
        worms = [_FakeWorm(root, n, json.loads(json.dumps(genome)))
                 for n in ("Alice", "Bob")]
        n_before = len(state.parent_vector)

        assert generations.migrate_genome_layout(state, worms) is True

        n_genes = len(lifelike.PARAM_SPEC)
        assert len(state.parent_vector) == n_before + n_genes
        assert len(state.parent_keys) == n_before + n_genes

        # The whole point: a worm's genome must still flatten to exactly the
        # parent's length, or the rollover raises mid-flight.
        for w in worms:
            vec, keys = evolution.flatten_weights(w.genome())
            assert len(vec) == len(state.parent_vector)
            assert [list(k) for k in keys] == state.parent_keys

        # Seeded at the current hardcoded defaults => behaviour is continuous.
        for i, name in enumerate(sorted(lifelike.PARAM_SPEC)):
            assert state.parent_vector[n_before + i] == lifelike.PARAM_SPEC[name][0]


def test_migration_is_idempotent_and_skips_the_control_arm():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        genome = _stock_genome()

        # flags off (poetry-4): never touched.
        state = _state_from(genome)
        worms = [_FakeWorm(root / "off", "Alice", json.loads(json.dumps(genome)))]
        n_before = len(state.parent_vector)
        assert generations.migrate_genome_layout(state, worms) is False
        assert len(state.parent_vector) == n_before

        # flags on: migrates once, then reports no further change.
        with _EnvOn():
            state = _state_from(genome)
            worms = [_FakeWorm(root / "on", "Alice", json.loads(json.dumps(genome)))]
            assert generations.migrate_genome_layout(state, worms) is True
            after = list(state.parent_vector)
            assert generations.migrate_genome_layout(state, worms) is False
            assert state.parent_vector == after


def test_live_children_get_zero_eps_on_the_new_dimensions():
    """The current generation explored zero distance along the new axes, so
    zero is the honest eps — and it makes the first rollover a no-op there
    rather than a spurious gradient kick."""
    with _EnvOn(), tempfile.TemporaryDirectory() as td:
        root = Path(td)
        genome = _stock_genome()
        d = len(evolution.flatten_weights(genome)[1])
        state = _state_from(genome, children={
            "Alice": {"eps": [0.5] * d, "is_elite": False},
            "Bob": {"eps": None, "is_elite": True},
        })
        worms = [_FakeWorm(root, n, json.loads(json.dumps(genome)))
                 for n in ("Alice", "Bob")]
        generations.migrate_genome_layout(state, worms)

        n_genes = len(lifelike.PARAM_SPEC)
        alice = state.children["Alice"]["eps"]
        assert len(alice) == d + n_genes
        assert alice[:d] == [0.5] * d, "recorded exploration was altered"
        assert alice[d:] == [0.0] * n_genes
        # None is left alone; the rollover's fallback reconstructs it as
        # (genome - parent)/step, which is zero on the new dims.
        assert state.children["Bob"]["eps"] is None


# --------------------------------------------------------------------------
# per-dimension scaling
# --------------------------------------------------------------------------

def test_scale_none_and_all_ones_are_bit_identical():
    """Every stock lineage must step exactly as it did before scaling
    existed."""
    rng_a = np.random.default_rng(4)
    rng_b = np.random.default_rng(4)
    parent = np.arange(20, dtype=np.float64)
    ones = np.ones(20)

    ca, ea = evolution.spawn_population(parent, 5, 0.3, rng_a, scale=None)
    cb, eb = evolution.spawn_population(parent, 5, 0.3, rng_b, scale=ones)
    for x, y in zip(ca, cb):
        assert np.array_equal(x, y)
    for x, y in zip(ea, eb):
        assert np.array_equal(x, y)

    scores = [1.0, 5.0, 3.0, 2.0, 4.0]
    ua = evolution.nes_update(parent, ea, scores, sigma=0.3, scale=None)
    ub = evolution.nes_update(parent, eb, scores, sigma=0.3, scale=ones)
    assert np.array_equal(ua, ub)


def test_each_gene_steps_by_the_same_fraction_of_its_own_range():
    """The reason for scaling at all: under one isotropic sigma, starve_gain
    (range 0..3) explores 6x more slowly than eta (range 0..0.5)."""
    names = sorted(lifelike.PARAM_SPEC)
    keys = [("ADAL", "ADAR")] + [(lifelike.LIFELIKE_KEY, n) for n in names]
    scale = np.array(lifelike.genome_scale(keys))
    parent = np.array([1.0] + [lifelike.PARAM_SPEC[n][0] for n in names])
    sigma = 0.02

    # A one-sigma perturbation, expressed as a fraction of each gene's range.
    fractions = []
    for i, n in enumerate(names, start=1):
        _d, lo, hi = lifelike.PARAM_SPEC[n]
        fractions.append(sigma * scale[i] / (hi - lo))
    assert max(fractions) - min(fractions) < 1e-12, \
        f"genes explore at different rates: {dict(zip(names, fractions))}"
    assert abs(fractions[0] - sigma) < 1e-12

    # Connectome weights keep their historical absolute step.
    assert scale[0] == 1.0

    # And the scaled spawn really does move a wide-range gene further.
    rng = np.random.default_rng(0)
    children, _ = evolution.spawn_population(parent, 200, sigma, rng, scale=scale)
    spread = np.std(np.array(children), axis=0)
    i_starve = 1 + names.index("starve_gain")
    i_eta = 1 + names.index("eta")
    assert spread[i_starve] > 5 * spread[i_eta], \
        "wide-range gene should move proportionally further in absolute units"


def test_scaled_trust_region_matches_the_sampling_radius():
    """The trust region caps |dtheta| at the radius the children were
    actually sampled at; with a scale vector that radius is sigma*||s||,
    not sigma*sqrt(d)."""
    names = sorted(lifelike.PARAM_SPEC)
    keys = [("ADAL", "ADAR")] * 50 + [(lifelike.LIFELIKE_KEY, n) for n in names]
    scale = np.array(lifelike.genome_scale(keys))
    parent = np.zeros(len(keys))
    rng = np.random.default_rng(1)
    _children, eps = evolution.spawn_population(parent, 8, 1.0, rng, scale=scale)
    # Enormous lr so the trust region is definitely the binding constraint.
    out = evolution.nes_update(parent, eps, list(range(8)), sigma=1.0,
                               lr=1e6, scale=scale)
    cap = evolution.TRUST_RADIUS * 1.0 * float(np.linalg.norm(scale))
    assert float(np.linalg.norm(out - parent)) <= cap * (1 + 1e-9)


def test_genome_scale_handles_json_pairs_from_state_file():
    """state.json stores [src, tgt] lists, not tuples."""
    keys = [["ADAL", "ADAR"], ["_lifelike", "starve_gain"]]
    assert lifelike.genome_scale(keys) == [1.0, 3.0]
