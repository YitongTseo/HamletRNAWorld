"""High-level driver: owns SimState + Integrator + Bonder, exposes step()."""
from __future__ import annotations

from .bonding import Bonder
from .integrator import Integrator
from .state import SimParams, SimState
from .strand import Strand, make_strands


class World:
    """Builds a SimState from raw sentence strings + matching tokenization
    + an embedder. The embedder is used once to produce contextual TOKEN
    vectors. After that the simulator runs on letter beads."""

    def __init__(
        self,
        sentences_raw: list[str],
        tokens_per_sentence: list[list[str]],
        embedder,
        params: SimParams | None = None,
        seed: int = 0,
    ):
        if not sentences_raw:
            raise ValueError("need at least one sentence")
        self.strands: list[Strand] = make_strands(
            sentences_raw, tokens_per_sentence, embedder
        )
        self.state = SimState.from_strands(self.strands, params)
        self.integrator = Integrator(self.state, seed=seed)
        self.bonder = Bonder(self.state, seed=seed + 1)

    def step(self, n: int = 1) -> None:
        s = self.state
        for _ in range(n):
            self.integrator.step()
            # Kinematic drag: held letters are pinned to their drag_targets
            # *after* the integrator runs, so velocity-based instability
            # can't accumulate. Backbone tension still propagates to the
            # rest of the chain through the normal force pass.
            if s.drag_indices.size > 0:
                s.pos[s.drag_indices] = s.drag_targets
                s.vel[s.drag_indices] = 0.0
            self.bonder.maybe_update()
