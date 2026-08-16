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
    nes_update(parent, eps_list, scores, sigma, lr, trust_radius) -> new parent
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
LEARNING_RATE = 1.0      # η in the NES update. Raised from 0.1 on 2026-08-13
                         # when the step stopped carrying a 1/σ factor — see
                         # nes_update. Under the corrected scaling this puts a
                         # typical step at ~17% of the sampling radius; the old
                         # 0.1 would have meant ~1.7%, i.e. barely moving.
TRUST_RADIUS = 0.5       # hard cap: |Δθ| ≤ TRUST_RADIUS · σ · √d, so the parent
                         # can never move further than the cloud of children it
                         # actually measured. 0.5 ≈ the mean-shift-per-generation
                         # that CMA-ES uses (1/√μ_eff for μ_eff≈4).
# --- shape of a mutation ----------------------------------------------------
#
# Degrees of freedom of the Student-t the children are drawn from. Set it to
# float("inf") for the plain Gaussian, which is what every generation before
# 2026-08-16 used and is how the A/B in tests/test_evolution.py is written.
# (Not None: the helpers take df=None to mean "use this default", and one
# sentinel doing both jobs silently turned an A/B into two identical runs.)
#
# WHY NOT GAUSSIAN. Real mutational effects are not bell-shaped. The measured
# distributions are L-shaped and heavy-tailed: in Chlamydomonas MA lines, 95%
# of mutations changed the expression of 0-1 genes and a 5% tail changed tens
# or hundreds, and a single variance parameter provably fails to describe them
# (higher moments differ between genes). Evolution mostly makes small
# adjustments and occasionally makes a large one; a Gaussian makes neither —
# it makes middling ones, always, in every coordinate at once.
#
# WHY IT DOESN'T BREAK THE ESTIMATOR (the claim I got wrong first time). The
# eps in the NES update is not "the perturbation", it is the SCORE of the
# search distribution, ∇_θ log p(x−θ). For an isotropic Gaussian that score
# happens to equal eps, which is why the two are usually written the same way.
# Swap the sampling law and the estimator stays exact provided you swap the
# score with it. For a Student-t the score is
#
#     s(u) = (ν+1)·u / (ν + u²)          u = the standardised variate
#
# which rises, peaks, and then REDESCENDS. So the heavy tail buys genuine
# large mutations while the update refuses to be hijacked by one: a child at
# 10σ contributes about a fifth of what a child at 1σ contributes. That is a
# redescending M-estimator, and it is exactly the property you want when a
# lucky judge score lands on a wild outlier.
#
# ν=3 rather than 1 (Cauchy, as in fast evolutionary programming): ν>2 keeps
# the variance finite, so σ still means what the σ-controllers and the trust
# region assume it means. The variates are rescaled to unit variance for the
# same reason. xNES's σ signal (‖eps‖²/d − 1) keeps its zero mean under that
# rescaling but gets noisier; the live arms run vs_mean, which is unaffected.
MUTATION_DF: float = 3.0

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

# Token-diversity measure. v7 originally multiplied this into `fitness` as a
# repetition PENALTY (the gardener kept flagging that the judge hands perfect
# scores to raw repetition — "God God God", "strew'd strew'd strew'd").
#
# 2026-08-13: REMOVED FROM THE FITNESS PATH. Repetition is now unpenalized —
# worms are free to repeat, and the judge's opinion stands unmodified. The
# function is kept because the judge-noise experiments use it as a
# stratification variable (experiments/judge_noise/nonllm_eval.py), not
# because anything in the live scoring path calls it.
REP_FLOOR_RATIO = 0.6   # unique-token ratio at/above which a window is undiscounted
REP_MIN_FACTOR = 0.05   # floor of the (no longer applied) discount


def repetition_factor(tokens: list[str]) -> float:
    """Token-diversity multiplier in [REP_MIN_FACTOR, 1.0]. Natural English
    (unique ratio ≳0.6, allowing repeated 'the'/'and') is unpenalized; heavy
    repetition is crushed toward the floor. Case-insensitive."""
    n = len(tokens)
    if n <= 1:
        return 1.0
    ratio = len({t.lower() for t in tokens}) / n
    # Squared below the floor so filler (ratio ~0.4-0.5) is crushed, not just
    # nicked; at/above the floor it saturates to 1.0 (no penalty).
    f = min(1.0, ratio / REP_FLOOR_RATIO) ** 2
    return max(REP_MIN_FACTOR, f)


# Divisor floor for the per-window mean. ≈ the window count of a typical
# surviving worm (~750 words → ~50 windows → 25% sampled ≈ 12), measured over
# data-lifelike-2 gens 1-6 where the flask median was 12-13. Below the floor
# the mean is diluted (sum/12), so a worm that dies after one lucky window
# cannot out-rank a full life of decent ones; above it, pure mean — extra
# volume buys nothing.
FITNESS_WINDOW_FLOOR = 12


def fitness(scored_windows: Iterable[ScoredWindow]) -> float:
    """MEAN over windows of  EMOTIONAL_WEIGHT * (E/100)^GAMMA + (C/100)^GAMMA,
    with the divisor floored at FITNESS_WINDOW_FLOOR.

    GAMMA makes top-rated windows dominate (a 90 contributes ~10× a 50); the
    1.5× weights emotional impact above coherence.

    2026-08-13: the repetition discount that v7 applied here is GONE — let them
    repeat. A repetitive window and a diverse one with identical E/C now score
    identically, so the judge's rating is the only thing that decides.

    2026-08-15 REGIME CHANGE (flask_1 from gen 8): was a plain SUM, which paid
    linearly for volume — sampled windows scale with words eaten, so a worm
    could out-rank better poets by eating more (data-lifelike-2 gen 6: the
    top eater ranked #2 on sum with the flask's second-worst per-window
    quality). Volume no longer needs a fitness incentive because hunger
    (WORMLET_HUNGER=1) enforces eating environmentally — starvation kills.
    The judge now pays for quality alone. NES selection is rank-based, so
    this is a pure re-ordering; σ and learning rate are untouched. Fitness
    values before/after this change are not comparable."""
    total = 0.0
    n = 0
    for w in scored_windows:
        e = (w.emotional / 100.0) ** GAMMA
        c = (w.coherence / 100.0) ** GAMMA
        total += EMOTIONAL_WEIGHT * e + 1.0 * c
        n += 1
    # n == 0 needs no special case: 0.0 / FLOOR == 0.0.
    return total / max(n, FITNESS_WINDOW_FLOOR)


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
    trust_radius: float = TRUST_RADIUS,
) -> np.ndarray:
    """One NES gradient step toward higher score, with the step size DECOUPLED
    from sigma via a trust region.

    Children were θ_i = parent + sigma * eps_i  with one elite at eps_0 = 0.

        θ' = parent + clip( lr * sigma * (1/N) * Σ_i rank_weight_i * eps_i )

    where clip bounds the step to  |Δθ| ≤ trust_radius * sigma * √d.

    WHY THE FORMULA CHANGED (2026-08-13)
    ------------------------------------
    This used to be  θ' = parent + (lr / (N * sigma)) * Σ rw_i eps_i, i.e. the
    step carried a 1/sigma factor. Σ rw_i eps_i has a magnitude that does not
    depend on sigma (the eps are unit normals), so that made

        |Δθ| ∝ 1/sigma        while the children were sampled at   sigma * √d.

    The two moved in OPPOSITE directions, so no value of sigma was ever sane
    and no sigma controller could rescue it. Measured on the live run:

        sigma=3.00 (gen 31)   children sampled at 182   parent stepped   0.34
        sigma=0.02 (gen 101)  children sampled at 1.2   parent stepped  52

    At high sigma the offspring are near-random and the parent is frozen; at
    low sigma the offspring are clones and the parent leaps ~43x beyond
    anything it measured, which is a noise-driven random walk — |θ| inflated
    253 → 420 over 101 generations. The only balance point was sigma ≈ 0.13,
    and the one arm parked near it (poetry-4 at sigma=0.1) was the only one
    whose fitness slope was positive and whose |θ| was stable.

    The 1/sigma was also simply the wrong gradient. For an isotropic Gaussian
    search distribution N(θ, sigma² I), the PLAIN gradient of expected fitness
    is (1/(N sigma)) Σ rw_i eps_i, but the NATURAL gradient — which is what the
    docstring always claimed and what "NES" means — preconditions by the
    inverse Fisher information, sigma² I, giving

        natural grad = sigma² * (1/(N sigma)) Σ rw_i eps_i
                     = (sigma / N) Σ rw_i eps_i.

    That is the form used here, and it makes |Δθ| ∝ sigma, i.e. proportional to
    the radius actually sampled. The ratio step/sampling-radius is now a
    constant ~lr * 0.17 for N=11, independent of sigma.

    The trust region on top is belt-and-braces: even if the rank-weighted sum
    comes out unusually large, the parent still cannot move outside the region
    its children explored, so the gradient is never extrapolated.

    Args:
        parent:       (d,) float vector
        eps_list:     list of (d,) float vectors, one per child (including
                      the elite, which is zeros)
        scores:       raw fitness per child, same order as eps_list
        sigma:        mutation strength used to spawn the children
        lr:           learning rate (η)
        trust_radius: cap on |Δθ| as a multiple of the sampling radius σ√d
    """
    n = len(scores)
    if n == 0 or n != len(eps_list):
        return parent.copy()
    if sigma <= 0.0:
        return parent.copy()

    # Sort indices by descending score → rank 0 = best.
    order = sorted(range(n), key=lambda i: -scores[i])
    rw = rank_weights(n)
    weighted_eps = np.zeros_like(parent)
    for rank, child_idx in enumerate(order):
        # The score of the search distribution, NOT the raw perturbation. They
        # are the same thing for a Gaussian; under the heavy-tailed default
        # (MUTATION_DF) the score redescends, so a freak child cannot drag the
        # parent out past the cloud it actually measured.
        weighted_eps += rw[rank] * mutation_score(eps_list[child_idx])

    # Natural-gradient step: magnitude scales WITH sigma.
    step = (lr * sigma / n) * weighted_eps

    # Trust region: never move further than the children were sampled.
    cap = trust_radius * sigma * np.sqrt(parent.shape[0])
    norm = float(np.linalg.norm(step))
    if norm > cap > 0.0:
        step *= cap / norm

    return parent + step


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
        eps = sample_eps(parent.shape, rng)
        eps_list.append(eps)
        children.append(parent + sigma * eps)
    return children, eps_list


def _t_scale(df: float) -> float:
    """Factor that rescales a raw t_ν variate to unit variance (Var = ν/(ν−2))."""
    return float(np.sqrt((df - 2.0) / df))


def sample_eps(shape, rng: np.random.Generator,
               df: float | None = None) -> np.ndarray:
    """One mutation vector: unit-variance, heavy-tailed by default.

    Unit variance is what keeps sigma meaning the same thing it meant under
    the Gaussian, so the sigma controllers and the trust region need no
    retuning (see MUTATION_DF)."""
    df = MUTATION_DF if df is None else df
    if not np.isfinite(df):
        return rng.standard_normal(shape)
    # MULTIVARIATE t: ONE chi-square scale for the whole child, not an
    # independent draw per coordinate. This is the difference between "every
    # synapse rolls its own dice" and "this individual carries a large-effect
    # mutation", and at this dimensionality it is the whole ball game.
    # Measured at d=3696, per-child ||eps||/sqrt(d):
    #
    #   Gaussian            p05 0.98  med 1.00  p99 1.03  max  1.04
    #   independent t3      p05 0.91  med 0.97  p99 1.36  max  2.90
    #   multivariate t3     p05 0.35  med 0.66  p99 3.20  max 29.5
    #
    # Concentration of measure eats per-coordinate tails: sum 3,696 of them
    # and every child comes out the same size, which is exactly the
    # middling-change-everywhere behaviour the Gaussian was rejected for.
    # With a shared scale, 21% of children carry a mutation >1.5x the median
    # and most carry less — the L-shape, at the level it is actually measured
    # in nature (an individual, not a coordinate).
    z = rng.standard_normal(shape)
    w = rng.chisquare(df) / df
    return _t_scale(df) * z / np.sqrt(w)


def mutation_score(eps: np.ndarray, df: float | None = None) -> np.ndarray:
    """∇_θ log p of the search distribution, in the same units as eps.

    Identity for the Gaussian — which is why the NES update could always be
    written in terms of eps directly — and redescending for the Student-t, so
    an outlier child informs the step less than a moderate one rather than
    more (MUTATION_DF)."""
    df = MUTATION_DF if df is None else df
    if not np.isfinite(df):
        return eps
    # Score of the MULTIVARIATE t: (nu+d)*u / (nu + ||u||^2), a per-child
    # scalar times the direction. It redescends in ||eps||, so the child that
    # gets down-weighted is the wildly-mutated INDIVIDUAL — the right unit,
    # since that is what the judge scored. Reduces to eps as nu -> inf, and to
    # ~eps for a typical child at finite nu.
    c = _t_scale(df)
    u = eps / c
    d = u.size
    return ((df + d) * u / (df + float(np.dot(u, u)))) / c


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
    fresh_mean: float | None = None      # mean fitness of fresh children this gen
                                         # (the parent-fitness proxy for next gen)
    scheme: str = "vs_mean"              # σ-control scheme used (Exp-2 A/B)
    sigma_baseline: float | None = None  # the incumbent the 1/5 rule compared to
                                         # (None for xnes / sigma_anneal)


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
    parent_fitness: float | None = None,
    prev_best_fitness: float | None = None,  # deprecated alias for parent_fitness
    scheme: str = "vs_mean",                 # Exp-2 σ-control A/B (σ-update only)
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

    # --- success rate → sigma (v7 σ-collapse fix) ---
    # v6 compared each fresh child against the previous generation's *best*
    # (max) score — the reigning champion. Almost no child beats the champion
    # under a noisy judge, so success_rate≈0 every gen and σ collapsed to the
    # floor and never recovered. v7 uses the Rechenberg-style POPULATION
    # BASELINE: compare against the parent's fitness, proxied by the PREVIOUS
    # generation's fresh-children mean (children are Gaussian samples around
    # the parent, so their mean ≈ the parent's fitness). This oscillates around
    # the 1/5 target instead of monotonically collapsing. `parent_fitness` is
    # the canonical arg; `prev_best_fitness` is accepted only for back-compat.
    fresh_mean = float(np.mean(fresh_scores)) if fresh_scores else None
    # Exp-2: the σ-control scheme is pluggable (σ-update ONLY — spawn/gradient
    # below are untouched). Default "vs_mean" reproduces the prior behaviour
    # exactly. Lazy import avoids the evolution<->sigma_controllers cycle.
    from server.sigma_controllers import GenContext, compute_new_sigma
    elite_fits = [fitnesses[i] for i in range(n) if is_elite[i]]
    ctx = GenContext(
        fresh_scores=fresh_scores,
        prev_fresh_mean=(parent_fitness if parent_fitness is not None else prev_best_fitness),
        prev_max=None,  # v6 control not wired for the live A/B
        elite_fitness=(max(elite_fits) if elite_fits else None),  # champion re-scored THIS gen
        centroid_fitness=None,  # vs_centroid needs an evaluated θ slot — follow-up
    )
    new_sigma, sr, sigma_baseline = compute_new_sigma(scheme, sigma, ctx, fresh_eps)
    success_rate = sr if sr is not None else SUCCESS_TARGET

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
        fresh_mean=fresh_mean,
        scheme=scheme,
        sigma_baseline=sigma_baseline,
    )
