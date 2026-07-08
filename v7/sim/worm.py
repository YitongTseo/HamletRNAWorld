"""Worm body: an inverse-kinematics chain trailing a steered head.

This mirrors worm-sim's main.js: the head is a `target` point that moves
forward at `speed` along `facing_dir`, and the IK chain's first link snaps
to the target each frame, with subsequent links lagging via spring relaxation.
The connectome's accum_left / accum_right drive turn rate and target speed.

Coordinate convention:
- World units are pixels (matching worm-sim) — we'll rescale on the viewer side.
- y-up. (worm-sim's main.js used screen coords with y-down: `target.y -= sin(...)`.
  We keep the same sign so the math is identical, then the viewer interprets
  the y axis however it likes.)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np


class IKChain:
    """A trailing inverse-kinematics chain, stored as parallel numpy arrays of
    per-segment head/tail coordinates and relaxed in ONE vectorized pass.

    Each segment's spring relaxation reads and writes ONLY its own head/tail —
    no neighbor coupling within the relax pass — so all N segments update
    independently in a handful of array ops instead of an N-iteration Python
    loop (~4x faster for N=200). The math (and its exact float64 evaluation
    order) mirrors the previous per-segment scalar code, so trajectories are
    BIT-IDENTICAL — determinism is preserved. `head[i]=tail[i-1]` propagation
    still happens first, using the previous tick's tails (a simple array shift).
    """

    def __init__(self, n_links: int, segment_size: float,
                 origin_x: float = 0.0, origin_y: float = 0.0,
                 facing: float | None = None,
                 rng: random.Random | None = None):
        """If `facing` is given (radians), lay the segments out in a straight
        line trailing behind the head along -facing. Otherwise jitter randomly
        using `rng` (defaults to an unseeded Random()). The straight layout is
        mainly useful for debug/test poses; in normal play the chain converges
        fast from any seed. The init draws `rng` in the SAME order as the old
        per-segment code so seeded worms keep identical initial poses."""
        self.size = float(segment_size)
        hx: list[float] = []
        hy: list[float] = []
        tx: list[float] = []
        ty: list[float] = []
        if facing is not None:
            # worm-sim's facing convention: target.y -= sin(facing)*speed.
            # So the trailing direction is (-cos(facing), +sin(facing)).
            dx = -math.cos(facing) * segment_size
            dy = +math.sin(facing) * segment_size
            x, y = origin_x, origin_y
            for _ in range(n_links):
                hx.append(x); hy.append(y)
                tx.append(x + dx); ty.append(y + dy)
                x, y = x + dx, y + dy
        else:
            r = rng if rng is not None else random.Random()
            x, y = origin_x, origin_y
            for _ in range(n_links):
                h_x = x + r.uniform(-1, 1)
                h_y = y + r.uniform(-1, 1)
                t_x = h_x + r.uniform(-1, 1)
                t_y = h_y + r.uniform(-1, 1)
                hx.append(h_x); hy.append(h_y); tx.append(t_x); ty.append(t_y)
                x, y = t_x, t_y
        self.hx = np.array(hx, dtype=np.float64)
        self.hy = np.array(hy, dtype=np.float64)
        self.tx = np.array(tx, dtype=np.float64)
        self.ty = np.array(ty, dtype=np.float64)

    def set_head(self, target_x: float, target_y: float) -> None:
        """Snap only the head to the target, deferring the relaxation. Used by
        the batched (per-flask) body path, where FlaskBody.relax() propagates +
        relaxes every worm's chain together in one numpy pass."""
        self.hx[0] = target_x
        self.hy[0] = target_y

    def _attach_row(self, hx, hy, tx, ty) -> None:
        """Repoint this chain at externally-owned row VIEWS (a FlaskBody's
        (W,N) arrays). In-place writes here and the batched relax then share
        memory — zero copy."""
        self.hx, self.hy, self.tx, self.ty = hx, hy, tx, ty

    def update(self, target_x: float, target_y: float) -> None:
        hx, hy, tx, ty = self.hx, self.hy, self.tx, self.ty
        # Snap the head to the target, then propagate: each segment's head
        # becomes the previous segment's (previous-tick) tail.
        hx[0] = target_x
        hy[0] = target_y
        hx[1:] = tx[:-1]
        hy[1:] = ty[:-1]
        # Spring relaxation — identical per-segment math, done for all N at once.
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

    def midline(self) -> list[tuple[float, float]]:
        """Return a polyline (head→tail) suitable for tube generation."""
        pts: list[tuple[float, float]] = [(float(self.hx[0]), float(self.hy[0]))]
        pts.extend(zip(self.tx.tolist(), self.ty.tolist()))
        return pts


@dataclass
class WormBody:
    n_segments: int = 200
    segment_size: float = 1.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    facing_dir: float = 0.0       # radians
    target_dir: float = 0.0
    speed: float = 0.0
    target_speed: float = 0.0
    speed_change: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    rng: random.Random | None = None
    # v7.1: when True, step() only snaps the head; a FlaskBody relaxes the whole
    # flask's chains together each tick. Set by FlaskBody.attach().
    batched: bool = False
    chain: IKChain = field(init=False)

    def __post_init__(self):
        self.target_x = self.origin_x
        self.target_y = self.origin_y
        self.chain = IKChain(self.n_segments, self.segment_size,
                             self.origin_x, self.origin_y, rng=self.rng)

    def consume_motor(self, accum_left: float, accum_right: float,
                      scaling_factor: float = 20.0) -> None:
        """Apply the connectome's L/R muscle accumulators to steering + speed.
        Same math as worm-sim's updateBrain()."""
        new_dir = (accum_left - accum_right) / scaling_factor
        self.target_dir = self.facing_dir + new_dir * math.pi
        self.target_speed = (abs(accum_left) + abs(accum_right)) / (scaling_factor * 5.0)
        self.speed_change = (self.target_speed - self.speed) / (scaling_factor * 1.5)

    def step(self) -> None:
        """One body-tick (worm-sim runs this at 60 Hz)."""
        self.speed += self.speed_change
        # Smallest signed angle from facing→target.
        diff = (self.target_dir - self.facing_dir + math.pi) % (2 * math.pi) - math.pi
        if diff > 0:
            self.facing_dir += 0.1
        elif diff < 0:
            self.facing_dir -= 0.1
        # worm-sim used `target.y -= sin(facing) * speed` (y-down screen
        # coords). We keep that sign and just let the viewer interpret y.
        self.target_x += math.cos(self.facing_dir) * self.speed
        self.target_y -= math.sin(self.facing_dir) * self.speed
        # Batched worms defer chain relaxation to their FlaskBody (one numpy
        # pass for the whole flask); standalone worms relax immediately.
        if self.batched:
            self.chain.set_head(self.target_x, self.target_y)
        else:
            self.chain.update(self.target_x, self.target_y)

    def midline(self) -> list[tuple[float, float]]:
        return self.chain.midline()
