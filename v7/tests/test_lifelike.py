"""Tests for lifelike mode (sim/lifelike.py): plasticity, hunger,
habituation, genome ride-along, determinism, and — most importantly — that
with all flags OFF nothing changes. No network, no LLM."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import lifelike
from sim.connectome import Connectome
from sim.world import World
from server.evolution import flatten_weights, unflatten_weights


def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return old


def _restore(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


ALL_OFF = dict(WORMLET_PLASTICITY=None, WORMLET_HUNGER=None,
               WORMLET_HABITUATION=None)
ALL_ON = dict(WORMLET_PLASTICITY="1", WORMLET_HUNGER="1",
              WORMLET_HABITUATION="1")
HAB_ONLY = dict(WORMLET_PLASTICITY=None, WORMLET_HUNGER=None,
                WORMLET_HABITUATION="1")


# --- params & genome ride-along --------------------------------------------

def test_pop_params_defaults_and_clipping():
    p = lifelike.pop_params(None)
    assert p["eta"] == 0.05 and p["trace_decay"] == 0.85
    assert p["adapt_rate"] == 0.05 and p["dishab_relief"] == 0.5
    w = {"_lifelike": {"eta": 99.0, "trace_decay": -3.0}, "AVAL": {"AVBL": 2}}
    p = lifelike.pop_params(w)
    assert p["eta"] == 0.5        # clipped to hi
    assert p["trace_decay"] == 0.0  # clipped to lo
    assert "_lifelike" not in w   # popped — Connectome never sees it
    assert "AVAL" in w


def test_params_flatten_through_evolution_unchanged():
    w = lifelike.ensure_params({"AVAL": {"AVBL": 2.0, "AVBR": -3.0}})
    vec, keys = flatten_weights(w)
    assert ("_lifelike", "eta") in keys  # rules are genes now
    back = unflatten_weights(vec, keys)
    assert back["_lifelike"]["eta"] == 0.05
    assert back["AVAL"]["AVBR"] == -3.0


# --- plasticity unit behaviour ---------------------------------------------

def _tiny_brain(plastic: bool) -> Connectome:
    # A -> B strong enough that firing A charges B past threshold next tick.
    c = Connectome(weights={"A": {"B": 40.0}, "B": {"A": 1.0}})
    if plastic:
        c.enable_plasticity(lifelike.pop_params(None))
    return c


def test_cofired_edge_gains_delta_on_reward():
    c = _tiny_brain(plastic=True)
    c.fire_neuron("A")
    c.fire_neuron("B")   # co-fired within one brain tick
    c.plasticity_step(reward=1.0)
    assert c._delta["A"]["B"] > 0.0
    assert c.delta_norm() > 0.0


def test_no_reward_no_delta_and_traces_decay_out():
    c = _tiny_brain(plastic=True)
    c.fire_neuron("A")
    c.fire_neuron("B")
    c.plasticity_step(reward=0.0)
    assert c._delta == {}
    for _ in range(200):     # trace_decay^200 << PRUNE_EPS
        c.plasticity_step(reward=0.0)
    assert c._trace == {}


def test_delta_decays_back_toward_genome():
    c = _tiny_brain(plastic=True)
    c.fire_neuron("A"); c.fire_neuron("B")
    c.plasticity_step(reward=5.0)
    d0 = c._delta["A"]["B"]
    for _ in range(50):
        c.plasticity_step(reward=0.0)
    d1 = c._delta.get("A", {}).get("B", 0.0)
    assert d1 < d0  # baseline_pull is forgetting


def test_inhibitory_edges_deepen_not_weaken():
    """Reinforcement must follow the genome's sign: an inhibitory synapse
    gets MORE inhibitory when its circuit leads to food (audit finding —
    unsigned increments eroded inhibition every meal)."""
    c = Connectome(weights={"A": {"B": -40.0}, "B": {"A": 1.0}})
    c.enable_plasticity(lifelike.pop_params(None))
    c.fire_neuron("A"); c.fire_neuron("B")
    c.plasticity_step(reward=1.0)
    assert c._delta["A"]["B"] < 0.0


def test_fresh_worm_dir_persists_lifelike_genes():
    """The rule params must be ON DISK in a fresh lineage's weights.json
    (rollovers flatten the disk genome) and must never be retro-injected
    into existing files (dimension mismatch would livelock the rollover)."""
    import json as _json
    import tempfile
    import server.orchestrator as orch
    old = _env(**ALL_ON)
    real = orch.FLASKS_DIR
    try:
        with tempfile.TemporaryDirectory() as td:
            orch.FLASKS_DIR = Path(td)
            wdir, w = orch._ensure_flask_worm_dir("flask_t", "wormy", 1)
            assert lifelike.LIFELIKE_KEY in w
            assert lifelike.LIFELIKE_KEY in _json.loads((wdir / "weights.json").read_text())
            # existing files are never retro-injected
            (wdir / "weights.json").write_text(_json.dumps({"A": {"B": 1.0}}))
            _, w2 = orch._ensure_flask_worm_dir("flask_t", "wormy", 1)
            assert lifelike.LIFELIKE_KEY not in w2
    finally:
        orch.FLASKS_DIR = real
        _restore(old)


def test_delta_capped():
    c = _tiny_brain(plastic=True)
    for _ in range(500):
        c.fire_neuron("A"); c.fire_neuron("B")
        c.plasticity_step(reward=100.0)
    assert c._delta["A"]["B"] <= lifelike.DELTA_CAP


def test_plasticity_stats_separates_learning_from_saturation():
    """delta_norm alone can't tell 'learned a lot' from 'hit the wall' (field
    case: three worms with the identical L1 of 29620.0 — 2,962 edges pinned at
    DELTA_CAP — one thriving, two starved). stats() must expose the split.
    baseline_pull=0 so deltas rest exactly AT the cap rather than pull-decayed
    a hair under it — the same phenotype the saturated field worms ran."""
    c = Connectome(weights={"A": {"B": 40.0}, "B": {"A": 1.0}})
    c.enable_plasticity({"eta": 0.05, "trace_decay": 0.85, "baseline_pull": 0.0,
                         "starve_gain": 0.8, "roam_gain": 0.5})
    c.fire_neuron("A"); c.fire_neuron("B")
    c.plasticity_step(reward=0.1)              # tiny nudge: learned, not capped
    s = c.plasticity_stats()                   # A->B and B->A both co-fired
    assert s["edges"] == 2 and s["capped"] == 0 and s["l1"] > 0
    for _ in range(500):
        c.fire_neuron("A"); c.fire_neuron("B")
        c.plasticity_step(reward=100.0)        # slam into the cap
    s = c.plasticity_stats()
    assert s["edges"] == 2 and s["capped"] == 2
    # Saturated l1 is the sum of the two edges' OWN caps, not 2xDELTA_CAP:
    # the bound is proportional now, so A->B (genome 40) may move by the
    # absolute ceiling 10 while B->A (genome 1.0) may move by 1.0.
    assert abs(s["l1"] - (lifelike.delta_cap_for(40.0)
                          + lifelike.delta_cap_for(1.0))) < 1e-6


def test_effective_weight_is_genome_plus_delta_and_genome_untouched():
    c = _tiny_brain(plastic=True)
    c.fire_neuron("A"); c.fire_neuron("B")
    c.plasticity_step(reward=1.0)
    d = c._delta["A"]["B"]
    c.dendrite_accumulate("A")
    assert abs(c.psyn["B"][c.next_state] - (40.0 + d)) < 1e-9
    assert c.weights["A"]["B"] == 40.0  # genome pristine — Darwinian


def test_plasticity_off_is_inert():
    c = _tiny_brain(plastic=False)
    c.fire_neuron("A"); c.fire_neuron("B")
    c.plasticity_step(reward=10.0)
    assert c._delta == {} and c._trace == {} and c._fired == set()


# --- hunger ------------------------------------------------------------------

def test_satiety_decays_and_death_freezes_worm():
    old = _env(**ALL_ON)
    try:
        w = World(seed=7)
        w.satiety = 3 * lifelike.SATIETY_DECAY_PER_TICK  # about to starve
        for _ in range(5):
            w.tick()
        assert w.dead and w.satiety == 0.0
        hx, hy, tc = w.worm.target_x, w.worm.target_y, w.tick_count
        for _ in range(60):
            w.tick()
        assert (w.worm.target_x, w.worm.target_y) == (hx, hy)  # corpse
        assert w.tick_count == tc + 60          # world/scroller kept going
    finally:
        _restore(old)


def test_death_record_captures_facts_at_death_and_freezes():
    """tick_count keeps advancing for a corpse, so the death tick / meal gap
    are only knowable at the moment of death — the record must capture them
    then and never change after (two worms starved with nothing on
    disk but dead=true; this record is the post-mortem)."""
    old = _env(**ALL_ON)
    try:
        w = World(seed=7)
        assert w.death_record is None
        w.satiety = 3 * lifelike.SATIETY_DECAY_PER_TICK  # about to starve
        for _ in range(5):
            w.tick()
        assert w.dead
        r = w.death_record
        assert r["cause"] == "starvation"
        assert r["last_meal_tick"] == -1                 # never ate...
        assert r["starved_ticks"] == r["died_at_tick"]   # ...fasted all life
        snap = dict(r)
        for _ in range(60):
            w.tick()
        assert w.death_record == snap                    # corpse ticks: frozen
    finally:
        _restore(old)


def test_death_record_checkpoint_roundtrip_keeps_logged_flag():
    """A restart must neither lose the post-mortem nor re-log it: the record
    (with the server's 'logged' marker) and last_meal_tick ride the lifelike
    checkpoint."""
    old = _env(**ALL_ON)
    try:
        w = World(seed=9)
        w._last_meal_tick = 0                # ate at birth, then nothing
        w.satiety = 2 * lifelike.SATIETY_DECAY_PER_TICK
        for _ in range(4):
            w.tick()
        assert w.dead and w.death_record is not None
        w.death_record["logged"] = True      # set by the server's death logger
        ck = w.lifelike_checkpoint()
        w2 = World(seed=9)
        w2.restore_lifelike(ck)
        assert w2.dead
        assert w2.death_record == w.death_record
        assert w2.death_record["logged"] is True
        assert w2._last_meal_tick == 0
    finally:
        _restore(old)


def test_eating_replenishes_satiety_and_rewards_brain():
    old = _env(**ALL_ON)
    try:
        w = World(seed=7)
        w.satiety = 0.5
        before = w.satiety
        # Simulate the eat branch directly (food placement is scroller-owned):
        w._recent_eaten.insert(0, "king")
        w._reward_accum += lifelike.REWARD_BASE + (1.0 - w.satiety)
        w.satiety = min(1.0, w.satiety + lifelike.SATIETY_BITE)
        assert w.satiety > before
        assert w._reward_accum == lifelike.REWARD_BASE + 0.5
    finally:
        _restore(old)


def test_starving_worm_smells_and_moves_harder():
    old = _env(**ALL_ON)
    try:
        w = World(seed=7)
        w.sensed_smells = {"k": {"neurons": {"ASEL": 0.2}}}
        w.satiety = 1.0
        fed = w._chemo_pulse()["ASEL"]
        w.satiety = 0.0
        starved = w._chemo_pulse()["ASEL"]
        assert starved > fed
        assert starved <= 1.0  # still saturates
    finally:
        _restore(old)


# --- habituation -------------------------------------------------------------

def test_static_smell_fades_then_recovers():
    """The core habituation loop: a constant smell loses its response (the
    baseline adapts up to it), silence recovers (baseline decays out), and
    the re-presented smell is back at full strength."""
    old = _env(**HAB_ONLY)
    try:
        w = World(seed=7)
        # Full smell shape — snapshot() below serialises it for the viewer.
        SMELL = {"k": {"word": "king", "x": 0.0, "y": 0.0, "distance": 10.0,
                       "pca": [0.0] * 12, "neurons": {"ASEL": 0.5}}}
        w.sensed_smells = dict(SMELL)
        first = w._chemo_pulse()["ASEL"]
        assert 0.0 < first < 0.5          # adapting from the first sniff
        for _ in range(300):              # (1-0.05)^300 ~ 2e-7: fully blind
            resp = w._chemo_pulse().get("ASEL", 0.0)
        assert resp < 0.01
        assert w.snapshot()["habituation"] > 0.4   # observable in the viewer
        w.sensed_smells = {}
        for _ in range(300):              # silence: baseline decays to prune
            w._chemo_pulse()
        assert w._adapt == {}
        w.sensed_smells = dict(SMELL)
        assert w._chemo_pulse()["ASEL"] == first   # full recovery
    finally:
        _restore(old)


def test_eating_dishabituates():
    old = _env(**HAB_ONLY)
    try:
        w = World(seed=7)
        w._adapt = {"ASEL": 0.4, "AWCL": 1e-5}
        w._dishabituate()
        assert abs(w._adapt["ASEL"] - 0.2) < 1e-12  # default relief = 0.5
        assert "AWCL" not in w._adapt               # dust pruned
        w.lifelike_params["dishab_relief"] = 1.0    # evolved to full reset
        w._dishabituate()
        assert w._adapt == {}
    finally:
        _restore(old)


def test_habituation_stacks_under_starvation_gain():
    """Adaptation is pre-gain (receptor level), hunger gain is downstream:
    a starving worm partially un-adapts, but a fully adapted channel stays
    silent no matter how hungry — gain multiplies (input - baseline), and
    0 times anything is 0."""
    old = _env(**ALL_ON)
    try:
        w = World(seed=7)
        w.sensed_smells = {"k": {"neurons": {"ASEL": 0.5}}}
        w.satiety = 1.0
        for _ in range(300):
            w._chemo_pulse()              # adapt fully while fed
        w.satiety = 0.0                   # now desperate
        assert w._chemo_pulse().get("ASEL", 0.0) < 0.01
    finally:
        _restore(old)


# --- checkpoint round-trip ---------------------------------------------------

def test_lifelike_checkpoint_roundtrip():
    """Restart must not cost the worm its within-life learning or its hunger:
    checkpoint -> fresh world -> restore == same satiety, deltas, traces."""
    import json
    old = _env(**ALL_ON)
    try:
        w = World(seed=9)
        w.satiety = 0.37
        # Real synapses, deltas inside their per-edge bound (AVAL->DA6 is 21,
        # AVAL->VA8 is 19): restore clamps to lifelike.delta_cap_for now, so
        # a test fixture on a synapse that doesn't exist — the old
        # AVAL->AVBL — would clamp to zero and read as a lost checkpoint.
        # That clamp is deliberate: plasticity retunes wiring, it cannot
        # invent a synapse the connectome doesn't have.
        w.brain._delta = {"AVAL": {"DA6": 2.5, "VA8": -1.25}}
        w.brain._trace = {("AVAL", "DA6"): 0.6}
        w._adapt = {"ASEL": 0.3, "AWCR": 0.05}
        ck = json.loads(json.dumps(w.lifelike_checkpoint()))  # via-JSON, like disk
        w2 = World(seed=9)
        w2.restore_lifelike(ck)
        assert w2.satiety == 0.37 and not w2.dead
        assert w2.brain._delta == {"AVAL": {"DA6": 2.5, "VA8": -1.25}}
        assert w2.brain._trace == {("AVAL", "DA6"): 0.6}
        # a restart must not dishabituate the nose for free
        assert w2._adapt == {"ASEL": 0.3, "AWCR": 0.05}
    finally:
        _restore(old)


def test_lifelike_checkpoint_none_when_off_and_v1_restore_tolerated():
    old = _env(**ALL_OFF)
    try:
        w = World(seed=9)
        assert w.lifelike_checkpoint() is None  # stock checkpoints don't grow
        w.restore_lifelike(None)                # v1 checkpoint: no-op
        w.restore_lifelike({"satiety": 0.1})    # features off: ignored
        assert w.satiety == lifelike.SATIETY_START
    finally:
        _restore(old)


# --- default-off equivalence & determinism ----------------------------------

def _run_world(ticks: int, seed: int = 11) -> tuple:
    w = World(seed=seed)
    for _ in range(ticks):
        w.tick()
    return (round(w.worm.target_x, 9), round(w.worm.target_y, 9),
            w.tick_count, tuple(w._recent_eaten))


def test_flags_off_no_lifelike_state_leaks():
    old = _env(**ALL_OFF)
    try:
        w = World(seed=3)
        for _ in range(240):
            w.tick()
        assert w.brain._delta == {} and w.brain._trace == {}
        assert w._adapt == {}
        assert not w.dead
        snap = w.snapshot()
        assert "satiety" not in snap and "plasticity_delta" not in snap
        assert "habituation" not in snap
    finally:
        _restore(old)


def test_deterministic_with_lifelike_on():
    old = _env(**ALL_ON)
    try:
        a = _run_world(400)
        b = _run_world(400)
        assert a == b
    finally:
        _restore(old)


def test_deterministic_with_lifelike_off_matches_itself():
    old = _env(**ALL_OFF)
    try:
        assert _run_world(400) == _run_world(400)
    finally:
        _restore(old)


# --- proportional delta cap -------------------------------------------------
#
# The flat DELTA_CAP was a blanket amplifier. Measured on the real connectome
# (3,689 edges, median |w| 2.0, p90 6.0): a uniform +10 multiplies the median
# synapse by 6 and the weakest by 11 while the strongest moves by 2.7, so a
# saturated worm's weights all end up at magnitude ~10-12 and the connectome
# stops carrying information. In the field (2026-08-16) five of ten worms in
# the beowulf flask had every plastic edge pinned and crawled in tight
# circles. Sustained circling is not wild-type C. elegans behaviour — it is
# the published phenotype of ablating the D-class GABAergic motor neurons,
# i.e. of wrecking the dorsoventral balance, which is what flattening every
# weight to one magnitude amounts to.

def _saturate(c: Connectome, reward=100.0, steps=500):
    for _ in range(steps):
        c.fire_neuron("A"); c.fire_neuron("B")
        c.plasticity_step(reward=reward)


def _eff(c: Connectome, pre: str, post: str) -> float:
    """Effective weight = genome + learned delta (Connectome applies the sum
    at dendrite_accumulate; there is no accessor, so spell it out here)."""
    return c.weights[pre][post] + c._delta.get(pre, {}).get(post, 0.0)


def _plastic(weights: dict) -> Connectome:
    c = Connectome(weights=weights)
    c.enable_plasticity({"eta": 0.05, "trace_decay": 0.85, "baseline_pull": 0.0,
                         "starve_gain": 0.8, "roam_gain": 0.5})
    return c


def test_a_synapse_can_at_most_double():
    """The proportional bound, stated as the thing it guarantees: learning
    scales a synapse, it never promotes a weak one into a strong one."""
    c = _plastic({"A": {"B": 1.5}, "B": {"A": 2.0}})
    _saturate(c)
    for pre, post, genome in (("A", "B", 1.5), ("B", "A", 2.0)):
        eff = abs(_eff(c, pre, post))
        assert eff <= 2 * genome + 1e-9, f"{pre}->{post} grew past double"
        assert eff > genome, "a reinforced synapse should still strengthen"


def test_saturation_preserves_the_connectomes_rank_order():
    """The failure mode in one assertion. Under the old flat cap a 1.0
    synapse saturated to 11.0 and a 30.0 synapse to 40.0 — a 30x difference
    compressed to under 4x, with the weak edge gaining more absolute drive
    than the strong one. Rank order has to survive learning."""
    c = _plastic({"A": {"B": 1.0}, "B": {"A": 30.0}})
    _saturate(c)
    weak = abs(_eff(c, "A", "B"))
    strong = abs(_eff(c, "B", "A"))
    assert strong / weak >= 15, f"weight structure flattened: {strong / weak:.1f}x"


def test_old_flat_cap_is_recoverable_and_is_the_thing_that_flattens():
    """Pins the A/B: DELTA_CAP_FRAC = inf IS the old rule. Kept executable so
    the claim above stays checkable rather than becoming folklore."""
    saved = lifelike.DELTA_CAP_FRAC
    try:
        lifelike.DELTA_CAP_FRAC = float("inf")
        c = _plastic({"A": {"B": 1.0}, "B": {"A": 30.0}})
        _saturate(c)
        weak = abs(_eff(c, "A", "B"))
        strong = abs(_eff(c, "B", "A"))
        assert abs(weak - 11.0) < 1e-6 and abs(strong - 40.0) < 1e-6
        assert strong / weak < 4, "the old rule flattened 30x down to under 4x"
    finally:
        lifelike.DELTA_CAP_FRAC = saved


def test_capped_metric_still_fires_under_the_proportional_bound():
    """plasticity_capped is the instrument that exposed the exploit. Compared
    against a flat DELTA_CAP it would now read 0 on a fully saturated worm —
    the metric would have gone blind exactly when it was needed."""
    c = _plastic({"A": {"B": 1.0}, "B": {"A": 2.0}})
    _saturate(c)
    s = c.plasticity_stats()
    assert s["edges"] == 2 and s["capped"] == 2
    assert s["l1"] < 2 * lifelike.DELTA_CAP, "l1 should be well under the flat cap"


def test_inhibition_still_deepens_and_never_flips():
    """Unchanged property, re-pinned next to the new bound: the delta carries
    the synapse's sign, so an inhibitory edge deepens and cannot cross zero."""
    c = _plastic({"A": {"B": -4.0}, "B": {"A": 1.0}})
    _saturate(c)
    assert _eff(c, "A", "B") < -4.0


def test_restored_checkpoint_deltas_are_clamped_to_the_new_bound():
    """Every checkpoint written before the proportional bound carries deltas
    clipped only by the old flat cap — a +10 on a |w|=1 synapse. Restoring
    one unclamped would put the brain straight back into the flattened,
    circling state until decay pulled it in."""
    old = _env(**ALL_ON)
    try:
        w = World(seed=3, weights={"A": {"B": 1.0}, "B": {"A": 30.0}})
        w.restore_lifelike({"delta": {"A": {"B": 10.0}, "B": {"A": 10.0}},
                            "trace": []})
        assert w.brain._delta["A"]["B"] == lifelike.delta_cap_for(1.0) == 1.0
        assert w.brain._delta["B"]["A"] == 10.0     # |w|=30: absolute ceiling
    finally:
        _restore(old)
