"""World: ticks the connectome (slow) and the body (fast), and tracks food.

The brain runs at ~2 Hz (worm-sim used setInterval(updateBrain, 500)).
The body runs at ~60 Hz. The world wraps both and exposes a snapshot dict
that the WebSocket server can ship to the viewer each frame.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from sim.connectome import Connectome
from sim.muscle_body import MuscleBody
from sim.worm import WormBody


# World extents in the same arbitrary "pixel" units as worm-sim. We pick a
# canvas so the worm has room to wander without immediately bouncing off walls.
WORLD_W = 1600.0
WORLD_H = 1000.0

# How often the brain updates (seconds).
BRAIN_PERIOD = 0.5

# Food parameters (worm-sim values).
FOOD_SENSE_RADIUS = 50.0
FOOD_EAT_RADIUS = 20.0
# How long a sensory stimulus lingers after the trigger fires (seconds).
STIM_LINGER = 2.0


@dataclass
class Food:
    x: float
    y: float


@dataclass
class World:
    """body_kind=`ik` (default) is worm-sim's kinematic IK chain — the head
    follows a steered "target" point, the body trails via spring relaxation.
    Looks smooth and worm-like.

    `muscle` integrates the midline directly from per-segment dorsal-ventral
    muscle differentials read from the connectome. Biologically grounded in
    principle, but the GoPiGo connectome we ported lacks proprioceptive
    feedback in the B-class motor neurons (Wen et al. 2012), so the muscle
    outputs jerk every brain tick (2 Hz) instead of producing a smooth
    travelling wave. Keep this kind here for when we wire up stretch
    feedback — the proper biological fix — and delete the CPG hack."""
    brain: Connectome = field(default_factory=Connectome)
    body_kind: str = "ik"
    worm: WormBody | MuscleBody = field(init=False)
    food: list[Food] = field(default_factory=list)

    # State flags, mirroring worm-sim.
    stim_hunger: bool = True
    stim_nose_touch: bool = False
    stim_food_sense: bool = False
    _stim_reset_at: float = 0.0
    _last_brain_tick: float = 0.0
    paused: bool = False

    def __post_init__(self):
        if self.body_kind == "muscle":
            self.worm = MuscleBody(
                origin_x=WORLD_W / 2, origin_y=WORLD_H / 2,
                body_length=800.0,
            )
        elif self.body_kind == "ik":
            self.worm = WormBody(
                n_segments=200, segment_size=4.0,
                origin_x=WORLD_W / 2, origin_y=WORLD_H / 2,
            )
        else:
            raise ValueError(f"unknown body_kind: {self.body_kind!r}")
        self.brain.rand_excite()

    def add_food(self, x: float, y: float) -> None:
        self.food.append(Food(x, y))

    def clear_food(self) -> None:
        self.food.clear()

    def _check_food(self) -> None:
        wx = self.worm.target_x
        wy = self.worm.target_y
        i = 0
        while i < len(self.food):
            f = self.food[i]
            d = math.hypot(wx - f.x, wy - f.y)
            if d <= FOOD_SENSE_RADIUS:
                self.stim_food_sense = True
                self._stim_reset_at = time.monotonic() + STIM_LINGER
                if d <= FOOD_EAT_RADIUS:
                    self.food.pop(i)
                    continue
            i += 1

    def _check_walls(self) -> None:
        if self.worm.target_x < 0:
            self.worm.target_x = 0
            self.stim_nose_touch = True
            self._stim_reset_at = time.monotonic() + STIM_LINGER
        elif self.worm.target_x > WORLD_W:
            self.worm.target_x = WORLD_W
            self.stim_nose_touch = True
            self._stim_reset_at = time.monotonic() + STIM_LINGER
        if self.worm.target_y < 0:
            self.worm.target_y = 0
            self.stim_nose_touch = True
            self._stim_reset_at = time.monotonic() + STIM_LINGER
        elif self.worm.target_y > WORLD_H:
            self.worm.target_y = WORLD_H
            self.stim_nose_touch = True
            self._stim_reset_at = time.monotonic() + STIM_LINGER

    def step(self, now: float) -> None:
        if self.paused:
            return
        # Brain at ~2 Hz.
        if now - self._last_brain_tick >= BRAIN_PERIOD:
            self.brain.tick(
                hunger=self.stim_hunger,
                nose_touch=self.stim_nose_touch,
                food_sense=self.stim_food_sense,
            )
            if isinstance(self.worm, WormBody):
                self.worm.consume_motor(self.brain.accum_left, self.brain.accum_right)
            self._last_brain_tick = now

        # Body at caller's frame rate. The muscle body re-reads per-muscle
        # activations every body tick (with smoothing); the IK body is driven
        # only at brain ticks via consume_motor() above.
        if isinstance(self.worm, MuscleBody):
            self.worm.consume(self.brain)
        self.worm.step()
        self._check_food()
        self._check_walls()

        # Decay transient stimuli back to "just hungry" after the linger window.
        if now >= self._stim_reset_at:
            self.stim_hunger = True
            self.stim_nose_touch = False
            self.stim_food_sense = False

    def snapshot(self) -> dict:
        """Serializable state for the viewer."""
        midline = self.worm.midline()
        return {
            "world": {"w": WORLD_W, "h": WORLD_H},
            "body_kind": self.body_kind,
            "midline": [[round(x, 2), round(y, 2)] for x, y in midline],
            "head": [round(self.worm.target_x, 2), round(self.worm.target_y, 2)],
            "facing": round(self.worm.facing_dir, 4),
            "speed": round(self.worm.speed, 4),
            "food": [[round(f.x, 2), round(f.y, 2)] for f in self.food],
            "motor": {"L": round(self.brain.accum_left, 2),
                      "R": round(self.brain.accum_right, 2)},
            "stim": {"hunger": self.stim_hunger,
                     "nose_touch": self.stim_nose_touch,
                     "food_sense": self.stim_food_sense},
            "paused": self.paused,
        }
