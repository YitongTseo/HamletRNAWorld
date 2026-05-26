"""Fitness aggregation + NES (Natural Evolution Strategies) update for the
generational evolution scheme.

Each worm's weights are a sparse adjacency dict
    {source_neuron: {target_neuron: int_weight, ...}, ...}
We flatten this to a numpy vector with a stable key ordering (sort by
(source, target) tuple), perturb each child with N(0, sigma), score them,
then update the parent in the direction the rank-weighted noise points.

Important quirk: live worm weights are INTEGERS (from the C. elegans wiring
database). NES does its math in float; we round when serializing back.
sigma needs to be large enough that rounding doesn't eat every mutation —
default is 0.5, not 0.02 as in the original spec draft. The user should
revisit this once they see how the first generation evolves.

Public API:
    flatten_weights(d)        -> (vec, keys)
    unflatten_weights(vec, k) -> dict (rounded to int)
    fitness(scored_windows)   -> float
    rank_weights(n)           -> ndarray
    nes_update(parent, eps_list, scores, sigma, eta=0.1) -> new parent vec
    adapt_sigma(sigma, improved) -> new sigma (clipped)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from server.judge import ScoredWindow

# Spec defaults; per-group state in generations.py can override.
SIGMA_INIT = 0.5         # larger than the spec's 0.02 because weights are int-rounded on write-back
SIGMA_MIN = 0.05
SIGMA_MAX = 3.0
SIGMA_SHRINK = 0.8       # multiply by this when generation improved
SIGMA_GROW = 1.3         # multiply by this when generation regressed
LEARNING_RATE = 0.1      # η in the NES update
GAMMA = 2.5              # fitness exponent — makes top windows dominate
EMOTIONAL_WEIGHT = 1.5   # vs 1.0 for coherence


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
    """Inverse of flatten_weights. Rounds to int — connectome weights are
    integer edge multiplicities, signed (negative = inhibitory)."""
    out: WeightDict = {}
    rounded = np.round(vec).astype(int)
    for (src, tgt), w in zip(keys, rounded):
        out.setdefault(src, {})[tgt] = int(w)
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


def adapt_sigma(sigma: float, improved: bool) -> float:
    """1/5-rule-style adaptation: shrink when the population is improving
    (refine the local optimum), grow when stagnating (escape it)."""
    factor = SIGMA_SHRINK if improved else SIGMA_GROW
    return float(np.clip(sigma * factor, SIGMA_MIN, SIGMA_MAX))


# --- helpers --------------------------------------------------------------

def spawn_children(parent: np.ndarray, n: int, sigma: float,
                   rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Returns (child_vectors, eps_vectors). child_vectors[0] is the elite
    (parent unchanged); child_vectors[1..n-1] are perturbed."""
    eps_list: list[np.ndarray] = [np.zeros_like(parent)]
    children: list[np.ndarray] = [parent.copy()]
    for _ in range(n - 1):
        eps = rng.standard_normal(parent.shape)
        eps_list.append(eps)
        children.append(parent + sigma * eps)
    return children, eps_list


@dataclass
class GenerationResult:
    sigma_used: float
    sigma_next: float
    best_score: float
    best_child_idx: int
    new_parent: np.ndarray
    ranks: list[int]  # ranks[i] = 0-based rank of child i (0 = best)
