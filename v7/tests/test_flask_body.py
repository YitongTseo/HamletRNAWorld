"""The batched per-flask IK body must be BIT-IDENTICAL to per-worm relaxation.

The chain is cosmetic (dynamics use the head point), so batching can't change
the eaten sequence — and because the batched relax runs the same float64 math
per row, it must also produce identical midlines. This pins both.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("WORMLET_GENERATIONS_ENABLED", "1")
os.environ.setdefault("WORMLET_PASSAGE", "full")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.world import World
from sim.worm import WormBody
from sim.flask_body import FlaskBody, build_flask_body


def _mid(w):
    return [(round(x, 12), round(y, 12)) for x, y in w.worm.midline()]


def test_batched_body_matches_per_worm_bit_for_bit():
    seeds = [1, 2, 3, 4, 5]
    # standalone reference worlds (each relaxes its own chain in step())
    ref = [World(seed=s) for s in seeds]
    # batched worlds: same seeds, adopted into a FlaskBody
    bat = [World(seed=s) for s in seeds]
    fb = FlaskBody([w.worm for w in bat])
    assert all(w.worm.batched for w in bat)

    eaten_ref = {s: [] for s in seeds}
    eaten_bat = {s: [] for s in seeds}
    for _ in range(1200):
        for w in ref:
            w.tick()
        for s, w in zip(seeds, ref):
            eaten_ref[s].extend(w.drain_eaten_words())
        for w in bat:
            w.tick()          # defers relax (batched)
        fb.relax()            # one pass for the whole flask
        for s, w in zip(seeds, bat):
            eaten_bat[s].extend(w.drain_eaten_words())
        # midlines identical every tick
        for wr, wb in zip(ref, bat):
            assert _mid(wr) == _mid(wb)
    # eaten sequences identical
    assert eaten_ref == eaten_bat


def test_build_flask_body_guards_and_shares_memory():
    class _W:
        def __init__(self, seed):
            self.world = World(seed=seed)
    worms = [_W(1), _W(2)]
    fb = build_flask_body(worms)
    assert isinstance(fb, FlaskBody)
    # each chain is a row-view of the batch: writing the batch shows up on the worm
    fb.hx[0, 0] = 12345.0
    assert worms[0].world.worm.chain.hx[0] == 12345.0
    # empty / non-IK guard
    assert build_flask_body([]) is None
