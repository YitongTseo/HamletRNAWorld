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

Genome integration: the evolvable rule genes live in the genome dict under
the reserved "_lifelike" pseudo-source, so evolution.py's flatten/unflatten
and the NES treat them exactly like synapse weights — the lineage searches
the LEARNING RULES, not just the wiring. World pops the block before the
dict reaches Connectome (which would otherwise mint a neuron named
"_lifelike"). Since 2026-08-16 the genes are stored in LOG (or logit)
coordinates and expressed as time constants, so the NES's additive step is a
multiplicative mutation — see GENE_SPEC for why, and for what the linear
version did to the population.

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

# --- the gene block: multiplicative, and expressed as time constants -------
#
# WHY THIS SHAPE (rewritten 2026-08-16, and the whole lineage restarted for
# it). The genes used to be plain numbers perturbed by the NES's single
# additive sigma, the same sigma that mutates synapse weights. Measured over
# 90 worm-generations of data-trio-3, that produced:
#
#     baseline_pull  54% of worms pinned at 0   -> no forgetting, ever
#     eta            33% pinned at 0            -> no plasticity at all
#     trace_decay    12% pinned at its ceiling
#
# from generation 1, in all three flasks. sigma was 0.149: fine against
# synapse weights (median |w| 2.0, a 7% jitter) and 7.5x the whole value of
# baseline_pull, whose default is 0.02. The rule genes were not evolving,
# they were being re-randomised every generation and clipped.
#
# Nature does not mutate a rate by adding a fixed quantity to it. Mutational
# effects on rates and expression levels are MULTIPLICATIVE — the effect is
# sampled and multiplied by the current value, so log(rate) performs an
# additive random walk and the trait distribution comes out log-normal
# (yeast promoter mutagenesis, PNAS 2019; evolutionary rate constants
# themselves span ~3 orders of magnitude within a lineage). So the genome
# now stores the LOG of each rate, and the NES's additive Gaussian step in
# that space is exactly a multiplicative step in the phenotype: one sigma
# means "mutate this rate by ~16%", whatever the rate's units. One sigma is
# correct for every gene once the genes are on the right scale, so no
# per-gene step sizes are needed.
#
# Two consequences worth stating plainly:
#   * Zero becomes unreachable. exp() never returns 0, so "never forgets"
#     and "cannot learn" stop being 54%- and 33%-probability accidents and
#     become impossible — which matches the animal: C. elegans forgetting
#     is an ACTIVE, gene-regulated process (TIR-1/JNK, MSI-1/Arp2/3,
#     dopamine via DOP-2/DOP-3 — dopamine-deficient mutants retain memories
#     LONGER), and there is no wild-type state in which potentiation
#     persists indefinitely without consolidation.
#   * Bounds stop being where the population lives. They are now far-field
#     sanity rails, hit by nothing.
#
# The fractions (dishab_relief) get the same treatment in logit space, so
# they stay strictly inside (0, 1) instead of piling up on an endpoint.
#
# TIME CONSTANTS, and where the numbers come from. Rates are easier to
# defend when written as "how long does this memory last", so the genes are
# time constants in SIM SECONDS and the per-tick rates are derived. Scaling
# the worm's real memory phases onto this sim: an adult C. elegans lives
# ~2 weeks and its short-term associative memory decays over ~1-2 h (one
# CS-US pairing), i.e. ~0.6% of adult life; long-term memory after spaced
# training lasts 12-24 h, ~7%. A worm here lives one generation, ~112 min,
# so the equivalents are ~40 s (STM) and ~470 s (LTM). The defaults below
# sit at the STM end and the bounds span both, which is the range a real
# lineage's memory genes actually cover.
#
# The phenotype defaults are within a few percent of the old hardcoded ones
# on purpose: same starting animal, different mutation structure.

import math

BRAIN_DT = 0.5                 # seconds per brain tick (2 Hz)

# genome key -> (transform, phenotype default, lo, hi). "log" genes are
# stored as log(value) and mutate multiplicatively; "logit" genes are stored
# as log(p/(1-p)) and stay strictly inside (0,1).
GENE_SPEC: dict[str, tuple[str, float, float, float]] = {
    # Learning rate. No animal has zero plasticity; the floor is small, not 0.
    "eta":            ("log",   0.05, 0.002, 0.5),
    # Eligibility trace: how long a co-firing mark stays consolidatable.
    # 3 s ~ the old trace_decay 0.85/tick, and the right order for a
    # dopamine-gated eligibility window.
    "tau_trace_s":    ("log",   3.0,  0.5,   30.0),
    # Forgetting. 40 s = the STM equivalent computed above (the old
    # baseline_pull 0.02/tick was 25 s, i.e. slightly faster than STM).
    # The ceiling is the LTM equivalent; nothing may forget slower.
    "tau_forget_s":   ("log",  40.0,  5.0,  600.0),
    "starve_gain":    ("log",   0.8,  0.05,   3.0),
    "roam_gain":      ("log",   0.5,  0.05,   3.0),
    # Habituation time constant (WORMLET_HABITUATION): 10 s reproduces the
    # old adapt_rate 0.05/tick. Short-term habituation in the real animal is
    # seconds-to-minutes, so the bounds bracket that.
    "tau_adapt_s":    ("log",  10.0,  1.0,  300.0),
    # Fraction of the adapted baseline a meal clears. A proportion, so logit.
    "dishab_relief":  ("logit", 0.5,  0.02,  0.98),
}

# What the sim actually consumes, derived from the genes above. Kept under
# the historical names so connectome.py, world.py, the death records and the
# viewer are all untouched by this change.
PARAM_KEYS = ("eta", "trace_decay", "baseline_pull", "starve_gain",
              "roam_gain", "adapt_rate", "dishab_relief")

# Keys that existed ONLY in the pre-2026-08-16 linear block. Their presence
# marks a genome from before the reparameterisation, whose numbers mean
# something different and must not be read as log coordinates.
LEGACY_GENE_KEYS = frozenset({"trace_decay", "baseline_pull", "adapt_rate"})

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


def _to_gene(name: str, value: float) -> float:
    """Phenotype -> genome coordinate (log or logit)."""
    kind, _d, lo, hi = GENE_SPEC[name]
    v = min(max(value, lo), hi)
    if kind == "log":
        return math.log(v)
    return math.log(v / (1.0 - v))          # logit


def _from_gene(name: str, g: float) -> float:
    """Genome coordinate -> phenotype, clamped to the gene's far-field rails.

    The clamp exists for arithmetic safety (a wild NES excursion shouldn't
    produce inf), not as a population boundary: at sigma 0.149 in log space a
    single mutation moves a rate ~16%, so the rails sit many mutations away
    from where anything lives. That is the whole point of the change — the
    old linear genes had HALF the population sitting on a bound."""
    kind, _d, lo, hi = GENE_SPEC[name]
    if kind == "log":
        v = math.exp(min(max(g, -50.0), 50.0))
    else:
        v = 1.0 / (1.0 + math.exp(-min(max(g, -50.0), 50.0)))
    return min(max(v, lo), hi)


def ensure_params(weights: dict) -> dict:
    """Add the _lifelike gene block (defaults, in GENOME coordinates) to a
    genome dict if missing. Mutates and returns the dict. Called at worm-dir
    load when the flags are on, so fresh lineages flatten the genes into the
    NES search space."""
    block = weights.setdefault(LIFELIKE_KEY, {})
    for name, (_kind, default, _lo, _hi) in GENE_SPEC.items():
        block.setdefault(name, _to_gene(name, default))
    return weights


def pop_params(weights: dict | None) -> dict[str, float]:
    """Remove the _lifelike block from a genome dict (so Connectome never
    sees it) and return the PHENOTYPE the sim consumes: per-brain-tick rates
    derived from the genes' time constants, under the historical key names.
    Safe on None and on dicts without the block.

    A pre-2026-08-16 block (linear genes: baseline_pull, trace_decay,
    adapt_rate) is NOT readable here — those keys mean something else now,
    and reading a stored 0.02 as log(tau) would give a 1-second memory. Such
    a block is ignored and the defaults are used; the lineages that carried
    them were restarted, which is why the keys were renamed rather than
    reused."""
    raw = (weights.pop(LIFELIKE_KEY, None) if weights is not None else None) or {}
    # Detected on the keys that exist ONLY in the old linear spec. "eta"
    # survived the rename (same meaning, log coordinate now), so its presence
    # proves nothing — a version marker would have been cleaner but every key
    # in this block is flattened into the NES vector and would get mutated.
    if raw and any(k in raw for k in LEGACY_GENE_KEYS):
        raw = {}
    genes = {name: float(raw[name]) if name in raw else _to_gene(name, default)
             for name, (_kind, default, _lo, _hi) in GENE_SPEC.items()}
    phen = {name: _from_gene(name, g) for name, g in genes.items()}

    # Time constants -> per-brain-tick rates. tau is the e-folding time, so
    # the per-tick fraction is 1 - exp(-dt/tau); with tau=40 s and dt=0.5 s
    # that is 0.0124, against the old hardcoded baseline_pull of 0.02.
    def rate(tau: float) -> float:
        return 1.0 - math.exp(-BRAIN_DT / tau)

    return {
        "eta":           phen["eta"],
        # Eligibility RETENTION per tick (the survivor of the decay), which
        # is what connectome.plasticity_step multiplies by.
        "trace_decay":   math.exp(-BRAIN_DT / phen["tau_trace_s"]),
        "baseline_pull": rate(phen["tau_forget_s"]),
        "starve_gain":   phen["starve_gain"],
        "roam_gain":     phen["roam_gain"],
        "adapt_rate":    rate(phen["tau_adapt_s"]),
        "dishab_relief": phen["dishab_relief"],
    }
