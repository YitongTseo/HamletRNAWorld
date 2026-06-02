"""Fitness aggregation + NES (Natural Evolution Strategies) update for the
generational evolution scheme.

Each worm's weights are a sparse adjacency dict
    {source_neuron: {target_neuron: int_weight, ...}, ...}
We flatten this to a numpy vector with a stable key ordering (sort by
(source, target) tuple), perturb each child with N(0, sigma), score them,
then update the parent in the direction the rank-weighted noise points.

The genome evolves in CONTINUOUS float space. The connectome weights start as
integers (from the C. elegans wiring database) but the sim consumes them as
floats, so we never round on write-back. Earlier versions rounded to int,
which ate the entire gradient — the parent moves ~0.05/weight/generation, far
below the 0.5 needed to flip an int — so nothing could be inherited. Keeping
float lets small mutations survive and accumulate.

Generation step (evolve_generation):
  - the NES gradient is estimated only from FRESH children, using their TRUE
    eps (persisted, not reconstructed from rounded weights);
  - the top-N genomes are carried VERBATIM into the next generation (elitism),
    the rest are fresh N(0, sigma) samples of the updated parent;
  - sigma adapts by Rechenberg's 1/5 success rule.

Public API:
    flatten_weights(d)        -> (vec, keys)
    unflatten_weights(vec, k) -> dict (float values)
    fitness(scored_windows)   -> float
    rank_weights(n)           -> ndarray
    nes_update(parent, eps_list, scores, sigma, eta=0.1) -> new parent vec
    adapt_sigma(sigma, success_rate) -> new sigma (clipped)
    spawn_population(parent, n, sigma, rng) -> (children, eps)
    evolve_generation(...) -> NextGen
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from server.judge import ScoredWindow

# Spec defaults; per-group state in generations.py can override.
# Change 1: the genome now evolves in CONTINUOUS float space (unflatten no
# longer rounds), so sigma no longer has to fight integer rounding. A small
# sigma now actually mutates the genome instead of being erased on write-back.
SIGMA_INIT = 0.1
SIGMA_MIN = 0.02
SIGMA_MAX = 3.0
SIGMA_SHRINK = 0.82      # multiply by this when success rate < 1/5 (steps too big)
SIGMA_GROW = 1.22        # multiply by this when success rate > 1/5 (steps too timid)
SUCCESS_TARGET = 0.2     # Rechenberg 1/5 rule target success fraction
LEARNING_RATE = 0.1      # η in the NES update
GAMMA = 1.5              # fitness exponent — lowered (was 2.5) so a worm that
                         # is consistently language-like beats one lucky window
EMOTIONAL_WEIGHT = 1.5   # vs 1.0 for coherence
N_ELITES = 5             # top-N genomes carried verbatim into the next gen


# --- weight flattening ---------------------------------------------------

WeightDict = dict[str, dict[str, int | float]]


def flatten_weights(weights: WeightDict) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Return (vector, keys) where keys[i] = (source, target) for vector[i].
    Stable order: sorted by (source, target) so the same weight dict always
    flattens to the same indices."""
    keys: list[tuple[str, str]] = []
    for src in sorted(weights):
        for tgt in sorted(weights[src]):
            keys.append((src, tgt))
    vec = np.array([weights[s][t] for s, t in keys], dtype=np.float64)
    return vec, keys


def unflatten_weights(vec: np.ndarray, keys: list[tuple[str, str]]) -> WeightDict:
    """Inverse of flatten_weights. Change 1: keeps FLOAT values — the genome
    evolves continuously. Connectome weights start as integer edge
    multiplicities (signed: negative = inhibitory) but the sim consumes them
    as floats (connectome.py: psyn[post] += w * scale), so fractional weights
    are valid. Rounding here was eating the entire NES gradient: the parent
    moves ~0.05/weight/generation, far below the 0.5 needed to flip an int,
    so the genome stayed frozen and nothing could be inherited."""
    out: WeightDict = {}
    for (src, tgt), w in zip(keys, vec):
        out.setdefault(src, {})[tgt] = float(w)
    return out


# --- fitness --------------------------------------------------------------

def fitness(scored_windows: Iterable[ScoredWindow]) -> float:
    """Sum over windows of  EMOTIONAL_WEIGHT * (E/100)^GAMMA + (C/100)^GAMMA.
    The exponent makes top-rated windows dominate (a 90 contributes ~10×
    more than a 50); the 1.5× weights emotional impact above coherence."""
    total = 0.0
    for w in scored_windows:
        e = (w.emotional / 100.0) ** GAMMA
        c = (w.coherence / 100.0) ** GAMMA
        total += EMOTIONAL_WEIGHT * e + 1.0 * c
    return total


# --- NES update -----------------------------------------------------------

def rank_weights(n: int) -> np.ndarray:
    """Standard NES rank weights: zero-mean, top-heavy.
    For n=10: roughly [4, 2.5, 1.5, 0.7, 0.2, -0.2, -0.7, -1.5, -2.5, -4]
    (computed as log((n+1)/2) - log(rank)). Sum is exactly 0."""
    ranks = np.arange(1, n + 1, dtype=np.float64)
    raw = np.maximum(0.0, np.log((n + 1) / 2.0) - np.log(ranks))
    raw -= raw.sum() / n  # zero-center
    return raw


def nes_update(
    parent: np.ndarray,
    eps_list: list[np.ndarray],
    scores: list[float],
    sigma: float,
    lr: float = LEARNING_RATE,
) -> np.ndarray:
    """One NES gradient step toward higher score.

    Children were θ_i = parent + sigma * eps_i  with one elite at eps_0 = 0.
    The score-weighted average of those noise vectors estimates the natural
    gradient of fitness; we step the parent that direction.

        θ' = parent + (lr / (N * sigma)) * Σ_i  rank_weight_i * eps_i

    Args:
        parent:    (d,) float vector
        eps_list:  list of (d,) float vectors, one per child (including
                   the elite, which is zeros)
        scores:    raw fitness per child, same order as eps_list
        sigma:     mutation strength used to spawn the children
        lr:        learning rate (η)
    """
    n = len(scores)
    if n == 0 or n != len(eps_list):
        return parent.copy()

    # Sort indices by descending score → rank 0 = best.
    order = sorted(range(n), key=lambda i: -scores[i])
    rw = rank_weights(n)
    weighted_eps = np.zeros_like(parent)
    for rank, child_idx in enumerate(order):
        weighted_eps += rw[rank] * eps_list[child_idx]

    return parent + (lr / (n * sigma)) * weighted_eps


def adapt_sigma(sigma: float, success_rate: float) -> float:
    """Change 4: Rechenberg's 1/5 success rule. `success_rate` is the
    fraction of fresh children that beat the incumbent. If we're succeeding
    more than 1/5 of the time the steps are too timid → grow sigma; less than
    1/5 means we're mostly missing → shrink; right at 1/5 → hold.

    This replaces the old single-noisy-best-score flag, which chased judge
    noise. A fraction over many children is far more stable."""
    if success_rate > SUCCESS_TARGET:
        factor = SIGMA_GROW
    elif success_rate < SUCCESS_TARGET:
        factor = SIGMA_SHRINK
    else:
        factor = 1.0
    return float(np.clip(sigma * factor, SIGMA_MIN, SIGMA_MAX))


# --- helpers --------------------------------------------------------------

def spawn_population(parent: np.ndarray, n: int, sigma: float,
                     rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Spawn `n` FRESH Gaussian samples of the search distribution.
    Returns (child_vectors, eps_vectors) where child[i] = parent + sigma*eps[i]
    and eps[i] ~ N(0, I). No elite-copy here — elitism is handled in
    evolve_generation by carrying real top-scoring genomes forward (Change 5),
    so every sample this function makes is a genuine perturbation we can use
    for the NES gradient estimate."""
    eps_list: list[np.ndarray] = []
    children: list[np.ndarray] = []
    for _ in range(n):
        eps = rng.standard_normal(parent.shape)
        eps_list.append(eps)
        children.append(parent + sigma * eps)
    return children, eps_list


@dataclass
class NextGen:
    """Result of one elitist-NES generation step."""
    new_parent: np.ndarray
    new_sigma: float
    next_genomes: list[np.ndarray]      # one per worm slot (elites first)
    next_epses: list[np.ndarray | None]  # aligned; None for carried elites
    next_is_elite: list[bool]            # aligned
    success_rate: float
    ranked_indices: list[int]            # current worms, best fitness first


def evolve_generation(
    parent_vec: np.ndarray,
    sigma: float,
    genomes: list[np.ndarray],
    epses: list[np.ndarray | None],
    is_elite: list[bool],
    fitnesses: list[float],
    *,
    n_elites: int,
    rng: np.random.Generator,
    prev_best_fitness: float | None = None,
) -> NextGen:
    """One generation of elitist NES (Changes 2 + 4 + 5).

    Args (all lists aligned, one entry per evaluated worm this generation):
        parent_vec:   current centroid θ
        sigma:        mutation strength used to spawn this generation
        genomes:      each worm's float genome as evaluated
        epses:        the TRUE Gaussian eps each fresh worm was spawned with
                      (Change 2 — no reconstruction from rounded weights);
                      None for carried elites / cold-start worms
        is_elite:     whether each worm was carried forward (excluded from grad)
        fitnesses:    judged fitness per worm
        n_elites:     how many top genomes to carry verbatim into the next gen
        prev_best_fitness: incumbent fitness, for the 1/5 success rate

    Returns the updated parent, adapted sigma, and the next generation's
    genomes (top-n_elites carried verbatim, the rest fresh NES samples).
    """
    n = len(genomes)

    # --- NES gradient from fresh children only (Change 2) ---
    # Elites are not Gaussian samples of THIS parent, so including them would
    # bias the gradient estimate; we use each fresh child's TRUE recorded eps.
    fresh_eps = [epses[i] for i in range(n) if not is_elite[i] and epses[i] is not None]
    fresh_scores = [fitnesses[i] for i in range(n) if not is_elite[i] and epses[i] is not None]
    if fresh_eps:
        new_parent = nes_update(parent_vec, fresh_eps, fresh_scores, sigma=sigma)
    else:
        new_parent = parent_vec.copy()

    # --- success rate → sigma (Change 4) ---
    if prev_best_fitness is None or not fresh_scores:
        success_rate = SUCCESS_TARGET  # neutral: hold sigma when we can't judge
    else:
        success_rate = sum(s > prev_best_fitness for s in fresh_scores) / len(fresh_scores)
    new_sigma = adapt_sigma(sigma, success_rate)

    # --- elitism: carry the top-n_elites genomes verbatim (Change 5) ---
    ranked = sorted(range(n), key=lambda i: -fitnesses[i])
    k = min(max(0, n_elites), n)
    elite_genomes = [genomes[i].copy() for i in ranked[:k]]

    # --- fill the rest with fresh samples of the UPDATED parent ---
    n_fresh = n - k
    if n_fresh > 0:
        fresh_children, fresh_child_eps = spawn_population(new_parent, n_fresh, new_sigma, rng)
    else:
        fresh_children, fresh_child_eps = [], []

    next_genomes = elite_genomes + fresh_children
    next_epses: list[np.ndarray | None] = [None] * k + fresh_child_eps
    next_is_elite = [True] * k + [False] * n_fresh

    return NextGen(
        new_parent=new_parent,
        new_sigma=new_sigma,
        next_genomes=next_genomes,
        next_epses=next_epses,
        next_is_elite=next_is_elite,
        success_rate=success_rate,
        ranked_indices=ranked,
    )
