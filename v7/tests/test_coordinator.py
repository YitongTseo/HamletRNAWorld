"""Tests for v7 shared-net co-evolution + cross-process coordinator."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import embedding
from server.shared_evolution import (
    pooled_update, spawn_perturbations, SharedNetState, SharedNetCoordinator,
)


def test_pooled_update_shape_and_determinism():
    d = embedding.GENOME_SIZE
    parent = np.zeros(d)
    rng = np.random.default_rng(0)
    genomes, epses = spawn_perturbations(parent, 0.1, 6, rng)
    scores = [1.0, 5.0, 2.0, 4.0, 3.0, 0.5]
    a = pooled_update(parent, epses, scores, 0.1, parent_fitness=2.0)
    b = pooled_update(parent, epses, scores, 0.1, parent_fitness=2.0)
    assert a.new_parent.shape == (d,)
    assert np.allclose(a.new_parent, b.new_parent)       # deterministic
    assert not np.allclose(a.new_parent, parent)          # it moved


def test_pooled_update_moves_toward_high_scorers():
    d = 8
    parent = np.zeros(d)
    # one clearly-best child pointing in +e0; gradient should have +e0 component
    eps = [np.eye(d)[0], -np.eye(d)[0], np.eye(d)[1], -np.eye(d)[1]]
    scores = [10.0, 0.0, 1.0, 1.0]
    upd = pooled_update(parent, eps, scores, 0.1, parent_fitness=1.0)
    assert upd.new_parent[0] > 0                           # stepped toward the winner
    assert upd.success_rate == 0.25                        # 1 of 4 beat baseline 1.0


def test_sigma_grows_and_shrinks():
    d = 4
    parent = np.zeros(d)
    eps = [np.random.default_rng(i).standard_normal(d) for i in range(10)]
    grow = pooled_update(parent, eps, [2.0]*9 + [0.0], 0.1, parent_fitness=1.0)
    shrink = pooled_update(parent, eps, [2.0] + [0.0]*9, 0.1, parent_fitness=1.0)
    assert grow.new_sigma > 0.1
    assert shrink.new_sigma < 0.1


def test_coordinator_all_flasks_report():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        flasks = ["flask_1", "flask_2", "flask_3"]
        coord = SharedNetCoordinator(root, flasks, timeout_s=5.0)
        d = embedding.GENOME_SIZE
        rng = np.random.default_rng(0)
        for f in flasks:
            _, epses = spawn_perturbations(np.zeros(d), 0.1, 4, rng)
            coord.contribute(0, f, [e.tolist() for e in epses], [1.0, 2.0, 3.0, 4.0])
        st = coord.barrier_update(0, now=lambda: 0.0, sleep=lambda s: None)
        assert st.generation == 1
        assert st.prev_fresh_mean is not None
        # published state persists and is loadable
        assert (root / "state.json").exists()
        assert (root / "gen-0000" / "done").exists()


def test_coordinator_quorum_on_partial_after_timeout():
    """A stalled flask must not freeze evolution — leader proceeds on quorum."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        flasks = ["flask_1", "flask_2", "flask_3"]
        coord = SharedNetCoordinator(root, flasks, timeout_s=1.0)
        d = embedding.GENOME_SIZE
        rng = np.random.default_rng(0)
        # only 2 of 3 report
        for f in ["flask_1", "flask_2"]:
            _, epses = spawn_perturbations(np.zeros(d), 0.1, 2, rng)
            coord.contribute(0, f, [e.tolist() for e in epses], [1.0, 2.0])
        # fake clock: jump straight past the timeout so we don't really sleep
        clock = {"t": 0.0}
        def now():
            clock["t"] += 0.5
            return clock["t"]
        st = coord.barrier_update(0, now=now, sleep=lambda s: None)
        assert st.generation == 1                     # proceeded despite missing flask_3


def test_coordinator_idempotent_when_done():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        coord = SharedNetCoordinator(root, ["flask_1"], timeout_s=1.0)
        d = embedding.GENOME_SIZE
        coord.contribute(0, "flask_1", [np.zeros(d).tolist()], [1.0])
        a = coord.barrier_update(0, now=lambda: 0.0, sleep=lambda s: None)
        gen_a = a.generation
        # second call sees `done` and just returns published state
        b = coord.barrier_update(0, now=lambda: 0.0, sleep=lambda s: None)
        assert b.generation == gen_a


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
