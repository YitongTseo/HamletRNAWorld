"""High-level driver: owns SimState + Integrator + Bonder, exposes step()."""
from __future__ import annotations

from .bonding import Bonder
from .integrator import Integrator
from .sequence import HAIRPIN_50
from .state import SimParams, SimState


# Default scene: two copies of the same hairpin, extruded sequentially through
# the same vent. Cheap way to demonstrate multi-chain support.
DEFAULT_SEQUENCES = [HAIRPIN_50, HAIRPIN_50]


class World:
    def __init__(
        self,
        sequences: str | list[str] | None = None,
        params: SimParams | None = None,
        seed: int = 0,
    ):
        if sequences is None:
            sequences = DEFAULT_SEQUENCES
        if isinstance(sequences, str):
            sequences = [sequences]
        self.state = SimState.from_sequences(sequences, params)
        self.integrator = Integrator(self.state, seed=seed)
        self.bonder = Bonder(self.state, seed=seed + 1)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.integrator.step()
            self.bonder.maybe_update()
