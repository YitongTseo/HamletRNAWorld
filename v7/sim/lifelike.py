"""Within-lifetime plasticity + hunger state ("lifelike" mode). Env-gated:

    WORMLET_PLASTICITY=1   reward-modulated Hebbian learning during life
    WORMLET_HUNGER=1       satiety state gating senses/motor + starvation

Both default OFF; with both flags off every code path is inert and the sim
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
}

# Hunger constants (fixed, not evolved — they define the environment, and an
# evolvable environment is a cheat the NES would find immediately).
SATIETY_START = 0.8            # born peckish, not starving
SATIETY_BITE = 0.12            # one word ~ one meal unit
SATIETY_DECAY_PER_TICK = 1.0 / 36000.0  # full -> starved in ~10 min sim time
REWARD_BASE = 0.5              # eating reward = REWARD_BASE + (1 - satiety)

# Learned deltas are bounded so plasticity can retune, not rewrite: |delta|
# stays within DELTA_CAP of the genome weight (genome weights are ~1..30).
DELTA_CAP = 10.0
# Traces/deltas below this are dropped to keep the sparse dicts sparse.
PRUNE_EPS = 1e-4


def plasticity_enabled() -> bool:
    return os.environ.get("WORMLET_PLASTICITY", "0") == "1"


def hunger_enabled() -> bool:
    return os.environ.get("WORMLET_HUNGER", "0") == "1"


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
