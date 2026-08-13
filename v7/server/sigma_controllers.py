"""Pluggable σ-control schemes for the Experiment-2 live A/B.

The bug (see docs/superpowers/specs/2026-07-17-...): the 1/5 success rule's
baseline must be the point the children were sampled AROUND, or its success
signal goes deaf to σ and σ rails. v6 compared to the champion MAX (success≈0 →
σ floors); v7 compares to the child MEAN (success≈0.5 → σ ceilings). Both are
order-statistics nearly constant in σ. This module makes the baseline pluggable
so we can A/B the correct choices against those controls.

Four of the five schemes are the SAME 1/5 rule with a different `baseline`:
  vs_max      previous generation's best (champion)      — v6 control (σ floors)
  vs_mean     previous generation's fresh-child mean      — v7 control (σ ceilings)
  vs_centroid the parent centroid θ's own fitness THIS gen (needs θ evaluated)
  vs_elite    the reigning elite's fitness THIS gen       ((1+λ) also spawns around it)
The fifth, `xnes`, drops the 1/5 rule entirely and adapts σ from the natural
gradient on step-size using the same ranked utilities — no baseline at all.

These are pure functions; wiring (evaluating θ, elite-centred spawning, per-flask
selection) lives in generations.py / evolution.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from server.evolution import adapt_sigma, SIGMA_MIN, SIGMA_MAX, SUCCESS_TARGET, rank_weights

ONE_FIFTH_SCHEMES = ("vs_max", "vs_mean", "vs_centroid", "vs_elite")
ALL_SCHEMES = ONE_FIFTH_SCHEMES + ("xnes",)


@dataclass
class GenContext:
    """Everything a σ-controller might need for one generation step."""
    fresh_scores: list[float]          # fitnesses of the fresh (non-elite) children
    prev_max: float | None = None      # last gen's best fitness            (vs_max)
    prev_fresh_mean: float | None = None  # last gen's fresh-child mean      (vs_mean)
    centroid_fitness: float | None = None  # θ scored THIS gen               (vs_centroid)
    elite_fitness: float | None = None     # reigning elite scored THIS gen  (vs_elite)


def _baseline(scheme: str, ctx: GenContext) -> float | None:
    return {
        "vs_max": ctx.prev_max,
        "vs_mean": ctx.prev_fresh_mean,
        "vs_centroid": ctx.centroid_fitness,
        "vs_elite": ctx.elite_fitness,
    }[scheme]


def success_rate(fresh_scores, baseline) -> float:
    """Fraction of fresh children strictly beating the baseline. Neutral
    (== SUCCESS_TARGET, i.e. hold σ) when we can't compute it."""
    if baseline is None or not fresh_scores:
        return SUCCESS_TARGET
    return sum(s > baseline for s in fresh_scores) / len(fresh_scores)


def one_fifth_step(scheme: str, sigma: float, ctx: GenContext) -> tuple[float, float]:
    """Return (new_sigma, success_rate) for a 1/5-rule scheme."""
    if scheme not in ONE_FIFTH_SCHEMES:
        raise ValueError(f"{scheme} is not a 1/5-rule scheme")
    sr = success_rate(ctx.fresh_scores, _baseline(scheme, ctx))
    return adapt_sigma(sigma, sr), sr


def sigma_anneal_step(sigma: float, decay: float = 0.8, floor: float = 0.1) -> float:
    """Deterministic geometric decay toward `floor` — no baseline, no signal.
    Pragmatic anti-ceiling control: whatever σ is (the live flasks are pinned at
    SIGMA_MAX), walk it back down to the small-σ regime where fitness was best
    (gen 1-2 ran at σ≈0.1). If forcing σ small restores learning, that alone
    confirms σ-runaway was the culprit."""
    return float(np.clip(max(floor, sigma * decay), SIGMA_MIN, SIGMA_MAX))


def compute_new_sigma(scheme: str, sigma: float, ctx: GenContext,
                      fresh_eps: list[np.ndarray] | None = None):
    """Single entry point the rollover calls. Returns (new_sigma, success_rate,
    baseline) — the latter two are None for the schemes that don't use a 1/5
    success signal, and are logged per generation so the A/B is readable."""
    if scheme == "xnes":
        return xnes_step(sigma, fresh_eps or [], ctx.fresh_scores), None, None
    if scheme == "sigma_anneal":
        return sigma_anneal_step(sigma), None, None
    baseline = _baseline(scheme, ctx)
    sr = success_rate(ctx.fresh_scores, baseline)
    return adapt_sigma(sigma, sr), sr, baseline


def xnes_step(sigma: float, eps_list: list[np.ndarray], scores: list[float],
              lr_sigma: float = 0.1) -> float:
    """Isotropic SNES/xNES step-size update — no baseline, no 1/5 rule.

    Children were θ_i = θ + σ·eps_i with eps_i ~ N(0, I) (dim d). With rank-based
    utilities u_i (top-heavy, mean 0), the natural-gradient signal for log σ is
        g = Σ_i u_i · (||eps_i||² / d − 1)
    and σ ← σ · exp(½ · lr · g), clipped. Utilities weight *better* children
    positively, so σ grows when the winners sit farther out than a unit sample
    and shrinks when the winners are the near-in ones — self-adapting toward the
    step that maximises expected progress, with no reference point to get wrong.
    """
    if not eps_list:
        return float(np.clip(sigma, SIGMA_MIN, SIGMA_MAX))
    n = len(eps_list)
    order = np.argsort(np.argsort([-s for s in scores]))  # 0 = best
    u = rank_weights(n)                                    # mean-0, top-heavy
    util = np.array([u[order[i]] for i in range(n)])
    d = eps_list[0].size
    g = sum(util[i] * (float(np.dot(eps_list[i], eps_list[i])) / d - 1.0) for i in range(n))
    return float(np.clip(sigma * np.exp(0.5 * lr_sigma * g), SIGMA_MIN, SIGMA_MAX))
