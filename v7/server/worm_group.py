"""WormGroup: a single 'flask' of N worms with its own NES lineage.

A flask is one independent evolutionary experiment. It owns:
  - N Worm objects (each with its own World, seed, weights)
  - A GenerationState (parent vector, sigma, generation counter, history)
  - Its own data subdirectory (data/flasks/<flask_name>/<worm_name>/...)
  - Its own generations archive (data/generations/<flask_name>/gen-NNNN/...)

End-of-corpus detection works at the flask level: a flask is considered
exhausted when all of its worms' text scrollers report corpus_exhausted.
Because all worms within a flask scroll Hamlet at the same fixed rate,
they exhaust simultaneously.

This module deliberately stays thin — the heavy machinery (rollover,
judging, NES, gardener) lives in `generations.py` and `gardener.py` and
takes a list of worms + a GenerationState. WormGroup is the holder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from server.orchestrator import Worm
from server.generations import GenerationState
from server.flask_embedder import FlaskEmbedderState
from server import embedding


@dataclass
class WormGroup:
    """One flask. The name is the directory-friendly identifier
    (e.g. 'flask_1'); the display label is for UI ('Flask 1')."""
    name: str
    display: str
    worms: list[Worm] = field(default_factory=list)
    state: GenerationState | None = None  # initialized lazily before first rollover
    # v7.1: this flask's OWN embedder lineage (independent of every other flask)
    # and the single shared EmbeddingModel all its worms sense through.
    embedder_state: FlaskEmbedderState | None = None
    embedding_model: "embedding.EmbeddingModel | None" = None
    # Wall-clock time the most recent corpus exhaustion was first detected;
    # used by sim_loop to enforce a brief grace period before firing rollover.
    corpus_exhausted_at: float | None = None

    @property
    def corpus_exhausted(self) -> bool:
        """True when every worm in this flask has run its TextScroller dry."""
        if not self.worms:
            return False
        return all(w.world.text_scroller.corpus_exhausted for w in self.worms)

    @property
    def tick_count(self) -> int:
        """Use the first worm as the flask's clock for the watchdog. All
        worms in a flask tick in lockstep, so any of them works."""
        return self.worms[0].world.tick_count if self.worms else 0
