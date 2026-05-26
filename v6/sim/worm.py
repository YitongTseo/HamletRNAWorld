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


@dataclass
class IKSegment:
    size: float
    head_x: float
    head_y: float
    tail_x: float
    tail_y: float

    def update(self) -> None:
        dx = self.head_x - self.tail_x
        dy = self.head_y - self.tail_y
        dist = math.sqrt(dx * dx + dy * dy) or 1e-9
        force = (0.5 - (self.size / dist) * 0.5) * 0.99
        strength = 0.998
        fx = force * dx
        fy = force * dy
        self.tail_x += fx * strength * 2.0
        self.tail_y += fy * strength * 2.0
        self.head_x -= fx * (1.0 - strength) * 2.0
        self.head_y -= fy * (1.0 - strength) * 2.0


class IKChain:
    def __init__(self, n_links: int, segment_size: float,
                 origin_x: float = 0.0, origin_y: float = 0.0,
                 facing: float | None = None,
                 rng: random.Random | None = None):
        """If `facing` is given (radians), lay the segments out in a straight
        line trailing behind the head along -facing. Otherwise jitter randomly
        using `rng` (defaults to an unseeded Random()). The straight layout is
        mainly useful for debug/test poses; in normal play the chain converges
        fast from any seed."""
        self.links: list[IKSegment] = []
        if facing is not None:
            # worm-sim's facing convention: target.y -= sin(facing)*speed.
            # So the trailing direction is (-cos(facing), +sin(facing)).
            dx = -math.cos(facing) * segment_size
            dy = +math.sin(facing) * segment_size
            x, y = origin_x, origin_y
            for _ in range(n_links):
                hx, hy = x, y
                tx, ty = x + dx, y + dy
                self.links.append(IKSegment(segment_size, hx, hy, tx, ty))
                x, y = tx, ty
        else:
            r = rng if rng is not None else random.Random()
            x, y = origin_x, origin_y
            for _ in range(n_links):
                hx = x + r.uniform(-1, 1)
                hy = y + r.uniform(-1, 1)
                tx = hx + r.uniform(-1, 1)
                ty = hy + r.uniform(-1, 1)
                self.links.append(IKSegment(segment_size, hx, hy, tx, ty))
                x, y = tx, ty

    def update(self, target_x: float, target_y: float) -> None:
        head = self.links[0]
        head.head_x = target_x
        head.head_y = target_y
        # First propagate the head position along the chain so each segment's
        # tail is the next segment's head, then run the spring relaxation.
        prev = head
        for link in self.links[1:]:
            link.head_x = prev.tail_x
            link.head_y = prev.tail_y
            prev = link
        for link in self.links:
            link.update()

    def midline(self) -> list[tuple[float, float]]:
        """Return a polyline (head→tail) suitable for tube generation."""
        pts = [(self.links[0].head_x, self.links[0].head_y)]
        for link in self.links:
            pts.append((link.tail_x, link.tail_y))
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
        self.chain.update(self.target_x, self.target_y)

    def midline(self) -> list[tuple[float, float]]:
        return self.chain.midline()
