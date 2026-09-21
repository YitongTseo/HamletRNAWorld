"""Common random numbers: every worm in a flask shares one seed per generation,
and that seed survives a restart.

Why this file exists. Measured on data-trio-4 gens 1-153 (3 flasks, 459
generation-observations), a genome the judge ranked top-5 last generation and
carried over VERBATIM as an elite beat a brand-new random mutant by Cohen's
d = -0.009, 95% CI [-0.080, +0.061]. Zero. The same null held for tokens eaten
(d = +0.068), which the judge never touches, so it was never only judge noise.

The cause was in here: the seed's entire influence on a life is
`Connectome.rand_excite()` — one draw of 40 random neurons kicked at birth —
plus the body's initial pose, and every worm got a DIFFERENT one. TextScroller
takes no rng, so worms already read an identical word stream; the per-worm seed
bought nothing and handed each genome its own random kick into a chaotic
300-neuron network, confounded with the genome in every comparison the NES made.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return old


def _restore(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# --- the seed round-trips through disk --------------------------------------

def test_seed_txt_is_read_back_not_recomputed():
    """seed.txt was WRITE-ONLY before 2026-09-01: _respawn_flask advanced the
    seed once per generation and persisted it, then every loader threw it away
    and recomputed the birth seed from the worm's index. The LCG chain reset on
    every restart, and a worm's SLOT — not its genome — permanently owned a
    birth kick."""
    import server.orchestrator as orch
    real = orch.FLASKS_DIR
    try:
        with tempfile.TemporaryDirectory() as td:
            orch.FLASKS_DIR = Path(td)
            wdir, _w, seed = orch._ensure_flask_worm_dir("flask_t", "wormy", 7)
            assert seed == 7, "first run takes the birth seed"
            assert (wdir / "seed.txt").read_text().strip() == "7"

            # A later generation advanced it, exactly as _respawn_flask does.
            (wdir / "seed.txt").write_text("123456789")
            _wdir2, _w2, seed2 = orch._ensure_flask_worm_dir("flask_t", "wormy", 7)
            assert seed2 == 123456789, (
                "a restart must resume the persisted seed, not recompute 7")
    finally:
        orch.FLASKS_DIR = real


def test_unreadable_seed_falls_back_to_the_birth_seed():
    """A truncated or garbage seed.txt must not take a flask down."""
    import server.orchestrator as orch
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "seed.txt"
        p.write_text("not-an-int")
        assert orch._read_seed(p, 42) == 42
        assert orch._read_seed(Path(td) / "absent.txt", 42) == 42


# --- the whole flask shares one seed per generation --------------------------

def _fake_flask(n_worms: int, seeds: list[int], tmp: Path):
    """Minimal stand-in for WormGroup: _respawn_flask only touches
    worms[].seed/.poem_path/.poem_file/.world/.word_count/.recent_words and
    flask.name/.state/.embedding_model."""
    class _W:
        def __init__(self, name, seed):
            self.name, self.seed = name, seed
            d = tmp / name
            d.mkdir(parents=True, exist_ok=True)
            self.poem_path = d / "poem.txt"
            self.poem_path.write_text("")
            self.poem_file = open(self.poem_path, "a", buffering=1)
            self.world = None
            self.word_count = 0
            self.recent_words = []

    class _S:
        generation = 3

    class _F:
        name = "flask_1"
        state = _S()
        embedding_model = None

    f = _F()
    f.worms = [_W(f"w{i}", s) for i, s in enumerate(seeds[:n_worms])]
    return f


def test_common_seed_gives_every_worm_in_a_flask_the_same_seed():
    import server.app as app
    old = _env(WORMLET_COMMON_SEED="1")
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            flask = _fake_flask(4, [1, 2, 3, 4], tmp)
            weights = {w.name: {"A": {"B": 1.0}} for w in flask.worms}
            app._respawn_flask(flask, weights)

            seeds = {w.seed for w in flask.worms}
            assert len(seeds) == 1, f"expected one shared seed, got {seeds}"
            # and it is persisted, so a restart resumes the shared value
            for w in flask.worms:
                on_disk = (w.poem_path.parent / "seed.txt").read_text().strip()
                assert int(on_disk) == w.seed
    finally:
        _restore(old)


def test_common_seed_still_changes_every_generation():
    """Shared, but not frozen — a fixed seed would overfit the lineage to one
    draw of the birth kick."""
    import server.app as app
    old = _env(WORMLET_COMMON_SEED="1")
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            flask = _fake_flask(3, [1, 2, 3], tmp)
            weights = {w.name: {"A": {"B": 1.0}} for w in flask.worms}
            app._respawn_flask(flask, weights)
            first = flask.worms[0].seed
            app._respawn_flask(flask, weights)
            assert flask.worms[0].seed != first
            assert len({w.seed for w in flask.worms}) == 1
    finally:
        _restore(old)


def test_common_seed_off_restores_per_worm_chains():
    """WORMLET_COMMON_SEED=0 is the A/B arm and must reproduce the old
    behaviour exactly: each worm advances its OWN chain."""
    import importlib
    import server.app as app
    old = _env(WORMLET_COMMON_SEED="0")
    try:
        importlib.reload(app)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            starts = [1, 2, 3]
            flask = _fake_flask(3, starts, tmp)
            weights = {w.name: {"A": {"B": 1.0}} for w in flask.worms}
            app._respawn_flask(flask, weights)
            got = [w.seed for w in flask.worms]
            assert got == [app._mutate_seed(s) for s in starts]
            assert len(set(got)) == 3, "per-worm chains must stay distinct"
    finally:
        _restore(old)
        importlib.reload(app)
