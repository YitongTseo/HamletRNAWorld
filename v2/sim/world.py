"""High-level driver: owns SimState + Integrator + Bonder, exposes step()."""
from __future__ import annotations

from .bonding import Bonder
from .integrator import Integrator
from .sequence import HAIRPIN_50
from .state import SimParams, SimState


class World:
    def __init__(
        self,
        sequence: str = HAIRPIN_50,
        params: SimParams | None = None,
        seed: int = 0,
    ):
        self.state = SimState.from_sequence(sequence, params)
        self.integrator = Integrator(self.state, seed=seed)
        self.bonder = Bonder(self.state, seed=seed + 1)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.integrator.step()
            self.bonder.maybe_update()
