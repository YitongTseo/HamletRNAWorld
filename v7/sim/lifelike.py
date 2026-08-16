"""Within-lifetime plasticity + hunger state ("lifelike" mode). Env-gated:

    WORMLET_PLASTICITY=1   reward-modulated Hebbian learning during life
    WORMLET_HUNGER=1       satiety state gating senses/motor + starvation
    WORMLET_HABITUATION=1  chemosensory adaptation — repeated smells fade

All default OFF; with every flag off every code path is inert and the sim
is bit-identical to stock v7 (test_lifelike.py asserts this).

Biology being imitated, and how loosely:
  * Plasticity — real C. elegans learns within a lifetime (habituation,
    food-odour association). Here: eligibility traces mark synapses whose
    pre AND post neuron co-fired recently; eating releases a reward pulse
    that consolidates traced synapses (dopamine-flavoured three-factor
    rule). Learned changes decay back toward the genome baseline.
  * Hunger — satiety drives roaming-vs-dwelling state switches (serotonin/
    dopamine in the real worm). Here: a [0,1] satiety level that starving
    pushes toward higher chemosensory gain (starved worms smell harder —
    measured in real worms) and higher motor gain (roaming), with death at
    zero. Eating while starved consolidates plasticity more strongly.
  * Habituation — the best-documented C. elegans learning: a repeated
    stimulus with no payoff loses its response (Rankin tap-withdrawal;
    Colbert & Bargmann olfactory adaptation), recovering once it stops,
    and a salient event (food) restores it. Here: each amphid neuron keeps
    an adapted baseline (EMA of its recent pre-gain input); the response is
    input minus baseline, floored at 0. A worm parked in a static smell
    field goes progressively nose-blind to it and leaves; fresh words
    upfield smell at full strength. Eating clears part of the baseline
    (dishabituation). Subtractive rather than divisive on purpose — the
    point is behavioural (leave the patch), not receptor fidelity, and
    subtraction makes departure decisive.

Genome integration: the five evolvable rule parameters live in the genome
dict under the reserved "_lifelike" pseudo-source, so evolution.py's
flatten/unflatten and the NES treat them exactly like synapse weights — the
lineage searches the LEARNING RULES, not just the wiring. World pops the
block before the dict reaches Connectome (which would otherwise mint a
neuron named "_lifelike").

Darwinian, not Lamarckian: learned weight changes live in Connectome._delta
and die with the worm. Rollovers read genomes from weights.json on disk
(generations.py:396), which never contains deltas — lifetime learning
affects fitness only, exactly like real life.

NOTE for existing lineages: parent_keys in a flask's state.json are fixed at
cold start. A lineage started before this feature carries no param genes —
worms run on the defaults below, which still works, but the rules themselves
only evolve in a lineage cold-started (fresh WORMLET_DATA_DIR) with the
flags on.
"""
from __future__ import annotations

import os

LIFELIKE_KEY = "_lifelike"

# name -> (default, lo, hi). Values are CLIPPED at point of use, not in the
# genome: NES perturbations (sigma ~0.1, tuned for synapse weights) may push
# a gene outside its meaningful range, and clipping at read time keeps the
# genome smooth for the gradient while the phenotype stays sane.
PARAM_SPEC: dict[str, tuple[float, float, float]] = {
    "eta":           (0.05, 0.0, 0.5),   # reward -> weight-change gain
    "trace_decay":   (0.85, 0.0, 0.99),  # eligibility memory (~4 brain ticks)
    "baseline_pull": (0.02, 0.0, 0.5),   # learned deltas relax to genome
    "starve_gain":   (0.8,  0.0, 3.0),   # extra chemosensory gain at S=0
    "roam_gain":     (0.5,  0.0, 3.0),   # extra motor gain at S=0
    # Habituation (WORMLET_HABITUATION). adapt_rate is the per-brain-tick EMA
    # rate of the adapted baseline: at 0.05 a static smell fades with a ~20
    # brain-tick (~10 s sim) time constant; 0 disables — evolution can switch
    # habituation off. dishab_relief is the fraction of the baseline cleared
    # per bite (1 = every meal fully resets the nose).
    "adapt_rate":    (0.05, 0.0, 0.5),
    "dishab_relief": (0.5,  0.0, 1.0),
}

# Hunger constants (fixed, not evolved — they define the environment, and an
# evolvable environment is a cheat the NES would find immediately).
SATIETY_START = 0.8            # born peckish, not starving
SATIETY_BITE = 0.12            # one word ~ one meal unit
SATIETY_DECAY_PER_TICK = 1.0 / 36000.0  # full -> starved in ~10 min sim time
REWARD_BASE = 0.5              # eating reward = REWARD_BASE + (1 - satiety)

# Learned deltas are bounded so plasticity can retune, not rewrite. TWO
# bounds, and the proportional one is the load-bearing half:
#
#   |delta| <= min(DELTA_CAP, DELTA_CAP_FRAC * |genome weight|)
#
# The absolute cap alone was a blanket amplifier. Measured on the real
# connectome (3,689 edges): median |w| is 2.0 and p90 is 6.0, so a uniform
# +10 multiplies the median synapse by 6 and the weakest by 11 while the
# strongest moves by 2.7. Every traced synapse lands at magnitude ~10-12
# whatever the biology said — the connectome's weight STRUCTURE is erased
# even though each delta keeps its synapse's sign. Observed in the field on
# 2026-08-16: five of ten worms in the beowulf flask had all 2,962 plastic
# edges pinned at the cap and crawled in tight circles. Sustained circling
# is not something a healthy C. elegans does; it is the published phenotype
# of ablating the D-class GABAergic motor neurons, i.e. of destroying the
# dorsoventral balance — which is what flattening every weight to the same
# magnitude amounts to.
#
# The proportional bound is also the more faithful rule: real synaptic
# plasticity scales a synapse's strength (LTP/LTD, synaptic scaling), it
# does not promote a weak synapse to the strongest in the circuit. At
# FRAC=1.0 a synapse may at most double, and the connectome's relative
# structure survives learning.
#
# DELTA_CAP_FRAC = inf reproduces the old absolute-only rule exactly, which
# is how the A/B in tests/test_lifelike.py is written.
DELTA_CAP = 10.0
DELTA_CAP_FRAC = 1.0


def delta_cap_for(weight: float) -> float:
    """Largest |delta| this synapse may accumulate."""
    return min(DELTA_CAP, DELTA_CAP_FRAC * abs(weight))
# Traces/deltas below this are dropped to keep the sparse dicts sparse.
PRUNE_EPS = 1e-4


def plasticity_enabled() -> bool:
    return os.environ.get("WORMLET_PLASTICITY", "0") == "1"


def hunger_enabled() -> bool:
    return os.environ.get("WORMLET_HUNGER", "0") == "1"


def habituation_enabled() -> bool:
    return os.environ.get("WORMLET_HABITUATION", "0") == "1"


def any_enabled() -> bool:
    """True when any lifelike feature is on — the gene-injection gate. Both
    cold-start sites (app._load_initial_default_weights and
    orchestrator._ensure_flask_worm_dir) call this, so adding a flag here is
    the ONLY edit that keeps their flatten dimensions in lockstep."""
    return plasticity_enabled() or hunger_enabled() or habituation_enabled()


def ensure_params(weights: dict) -> dict:
    """Add the _lifelike gene block (defaults) to a genome dict if missing.
    Mutates and returns the dict. Called at worm-dir load when the flags are
    on, so fresh lineages flatten the params into the NES search space."""
    block = weights.setdefault(LIFELIKE_KEY, {})
    for name, (default, _lo, _hi) in PARAM_SPEC.items():
        block.setdefault(name, default)
    return weights


def pop_params(weights: dict | None) -> dict[str, float]:
    """Remove the _lifelike block from a genome dict (so Connectome never
    sees it) and return CLIPPED params, defaults where absent. Safe on None
    and on dicts without the block."""
    raw = (weights.pop(LIFELIKE_KEY, None) if weights is not None else None) or {}
    out: dict[str, float] = {}
    for name, (default, lo, hi) in PARAM_SPEC.items():
        v = float(raw.get(name, default))
        out[name] = lo if v < lo else hi if v > hi else v
    return out
