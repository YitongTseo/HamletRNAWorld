"""Per-flask embedder evolution (v7.1).

ONE embedding+POS genome per flask, shared by every worm in the flask and held
FIXED within a generation, evolved by a (1+1)-ES with Rechenberg's 1/5 success
rule. No per-worm perturbation, no cross-process coordination — each flask
hill-climbs its own embedder independently, so N flasks explore N divergent,
hyper-specialized embedders in parallel, and no flask ever waits on another.

Why (1+1) and not the pooled per-worm NES the brains use: sharing the
precomputed E-table across a flask REQUIRES an identical `W_E` across the
flask's worms within a generation, which precludes a per-worm gradient for the
embedder. So the embedder evolves at flask granularity. Each generation the
flask senses with ONE candidate `θ + σ·eps`; its score is the MEAN of the
flask's worm fitnesses — the candidate is thus evaluated across all ~16 brains,
so the accept/reject decision is low-variance even from a single sample.

Upgrade path if a stronger gradient is wanted later: antithetic 2-point (split
the flask into `θ+σ·eps` and `θ−σ·eps` halves, gradient ∝ `(f₊−f₋)·eps`) — two
precomputed tables, still cheap.

State is persisted to `data/generations/<flask>/embedder.json`, alongside the
brain's `state.json`, so a restart resumes the same embedder lineage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from server import embedding
from server.evolution import adapt_sigma

# The embedder's own σ schedule. Starts a touch smaller than the brain's since
# the genome is ~7.7k densely-coupled net weights (a large step saturates the
# sigmoids). adapt_sigma() clips to the shared [SIGMA_MIN, SIGMA_MAX].
EMB_SIGMA_INIT = 0.08
# Window (in generations) over which the 1/5 success fraction is measured for σ
# adaptation — long enough to smooth judge noise, short enough to react.
EMB_SIGMA_WINDOW = 10


def flask_eps(seed: int, generation: int) -> np.ndarray:
    """Deterministic per-(flask, generation) Gaussian perturbation of the shared
    genome. Same (seed, generation) always yields the same eps, so a restart
    mid-lineage reconstructs the identical candidate."""
    s = (int(seed) * 2_654_435_761 + int(generation) * 40_503 + 1) % (2**32)
    return np.random.default_rng(s).standard_normal(embedding.GENOME_SIZE)


@dataclass
class FlaskEmbedderState:
    """One flask's embedder lineage under the (1+1)-ES.

    - `theta`: the incumbent genome (best accepted so far).
    - `candidate`: the genome CURRENTLY deployed to the flask's worms this
      generation (`theta + σ·eps`, or `theta` itself at generation 0 to measure
      the incumbent's own fitness).
    - `incumbent_score`: the incumbent's flask-mean fitness (None until the gen-0
      baseline is measured).
    """
    name: str
    seed: int = 0
    generation: int = 0
    sigma: float = EMB_SIGMA_INIT
    theta: list[float] | None = None
    candidate: list[float] | None = None
    incumbent_score: float | None = None
    accept_history: list[int] = field(default_factory=list)   # 1 = accepted
    score_history: list[float] = field(default_factory=list)

    @classmethod
    def load_or_init(cls, path: Path, name: str, seed: int = 0) -> "FlaskEmbedderState":
        if path.exists():
            data = json.loads(path.read_text())
            data.setdefault("name", name)
            data.setdefault("seed", seed)
            return cls(**data)
        theta = embedding.EmbeddingParams.random_init(seed).flatten().tolist()
        # Generation 0 deploys the incumbent itself (eps = 0) so we measure the
        # cold-start genome's fitness before proposing any perturbation.
        return cls(name=name, seed=seed, theta=theta, candidate=list(theta))

    def theta_vec(self) -> np.ndarray:
        return np.array(self.theta, dtype=np.float64)

    def candidate_vec(self) -> np.ndarray:
        return np.array(self.candidate, dtype=np.float64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self)))
        tmp.replace(path)


def step(state: FlaskEmbedderState, candidate_score: float) -> FlaskEmbedderState:
    """Advance the (1+1)-ES one generation given the deployed candidate's
    flask-mean fitness. Mutates and returns `state`; the caller then rebuilds
    the flask's shared EmbeddingModel from `state.candidate` (the genome for the
    NEXT generation).

    - Generation 0 establishes the incumbent baseline (candidate == incumbent).
    - Later generations accept the candidate iff it beats the incumbent, then
      adapt σ by the 1/5 rule over a rolling window and draw the next candidate.
    """
    if state.incumbent_score is None:
        # Baseline generation: the candidate WAS the incumbent (eps = 0).
        state.incumbent_score = float(candidate_score)
        accepted = 1
    elif candidate_score > state.incumbent_score:
        state.theta = list(state.candidate)          # accept: move the incumbent
        state.incumbent_score = float(candidate_score)
        accepted = 1
    else:
        accepted = 0                                  # reject: keep the incumbent

    state.accept_history.append(accepted)
    state.score_history.append(float(candidate_score))

    # σ ← 1/5 rule over the recent window (bolder if we accept often).
    window = state.accept_history[-EMB_SIGMA_WINDOW:]
    success_rate = sum(window) / len(window)
    state.sigma = adapt_sigma(state.sigma, success_rate)

    # Draw the next generation's candidate from the (possibly moved) incumbent.
    state.generation += 1
    eps = flask_eps(state.seed, state.generation)
    state.candidate = (state.theta_vec() + state.sigma * eps).tolist()
    return state


def build_model(state: FlaskEmbedderState) -> "embedding.EmbeddingModel":
    """The flask's SHARED EmbeddingModel for the currently-deployed candidate.
    Primes the E-table once; all the flask's worms sense through this one
    instance."""
    return embedding.EmbeddingModel(
        embedding.EmbeddingParams.from_flat(state.candidate_vec())
    )
