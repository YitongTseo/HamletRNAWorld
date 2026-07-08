"""Batched IK body relaxation for a whole flask (v7.1).

The IK chain is the single biggest per-tick compute, and it is **cosmetic** —
the simulation dynamics (`_check_food`, `_check_walls`, `_compute_smells`) all
read the head point (`target_x/y`), never the trailing chain; only the viewer's
`midline()` does. So the relaxation of every worm in a flask can be done together
in ONE numpy pass over stacked `(W, N)` arrays instead of W separate `(N,)`
passes (~4.5× faster per worm for W=16, N=200).

FlaskBody owns the `(W, N)` head/tail arrays and repoints each `WormBody`'s
`IKChain` at a row VIEW of them, so per-worm head writes (`set_head`, done in
`WormBody.step`) and the batched relax share memory with zero copying. The math
and its float64 evaluation order match `IKChain.update` exactly, so the produced
midlines are BIT-IDENTICAL to the per-worm path — determinism preserved.
"""
from __future__ import annotations

import numpy as np

from sim.worm import WormBody


class FlaskBody:
    """Holds one flask's IK chains as `(W, N)` arrays and relaxes them together.

    Build it AFTER the flask's worlds exist; it adopts their chains (stacking
    the current per-worm arrays once) and sets each worm to `batched=True`. Call
    `relax()` once per tick, after every worm in the flask has ticked (each has
    then snapped its own head via `WormBody.step` -> `IKChain.set_head`)."""

    def __init__(self, bodies: list[WormBody]):
        self.bodies = bodies
        self.size = bodies[0].chain.size
        # Stack the per-worm chain arrays into contiguous (W, N) buffers, then
        # hand each chain a row-view back so the two stay in sync with no copy.
        self.hx = np.stack([b.chain.hx for b in bodies])
        self.hy = np.stack([b.chain.hy for b in bodies])
        self.tx = np.stack([b.chain.tx for b in bodies])
        self.ty = np.stack([b.chain.ty for b in bodies])
        for i, b in enumerate(bodies):
            b.chain._attach_row(self.hx[i], self.hy[i], self.tx[i], self.ty[i])
            b.batched = True

    def relax(self) -> None:
        """One vectorized spring-relaxation pass for every worm in the flask.
        Heads (row 0) were already snapped to each worm's target by step();
        here we propagate (head[i]=tail[i-1]) and relax all segments at once."""
        hx, hy, tx, ty = self.hx, self.hy, self.tx, self.ty
        hx[:, 1:] = tx[:, :-1]
        hy[:, 1:] = ty[:, :-1]
        dx = hx - tx
        dy = hy - ty
        dist = np.sqrt(dx * dx + dy * dy)
        dist[dist == 0.0] = 1e-9              # mirrors the scalar `... or 1e-9`
        force = (0.5 - (self.size / dist) * 0.5) * 0.99
        strength = 0.998
        fx = force * dx
        fy = force * dy
        tx += fx * strength * 2.0
        ty += fy * strength * 2.0
        hx -= fx * (1.0 - strength) * 2.0
        hy -= fy * (1.0 - strength) * 2.0


def build_flask_body(worms) -> FlaskBody | None:
    """Build a FlaskBody for a flask's worms, or None if they aren't all
    standard IK bodies of the same segment count (e.g. MuscleBody) — in which
    case each worm keeps relaxing its own chain (correct, just unbatched)."""
    bodies = [w.world.worm for w in worms]
    if not bodies or not all(isinstance(b, WormBody) for b in bodies):
        return None
    n = bodies[0].chain.hx.shape[0]
    if any(b.chain.hx.shape[0] != n for b in bodies):
        return None
    return FlaskBody(bodies)
