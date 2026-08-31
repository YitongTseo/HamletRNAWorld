"""C. elegans connectome ported from heyseth/worm-sim (which ported it from
the GoPiGo Python original by Busbice/Garrett/Churchill).

Keeps the original integrate-and-fire semantics:
- Each neuron has a double-buffered accumulator [this_state, next_state].
- dendrite_accumulate(pre) adds pre's outgoing weights into the next_state
  bucket of each post-synaptic target.
- run() fires every non-muscle neuron whose this_state exceeds the fire
  threshold (firing = dendrite_accumulate(self) + zero own next_state),
  collects muscle activations into accum_left/accum_right, then copies
  next_state into this_state and swaps the buffers.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

WEIGHTS_PATH = Path(__file__).parent / "weights.json"

MUSCLE_PREFIXES = ("MVU", "MVL", "MDL", "MVR", "MDR")
FIRE_THRESHOLD = 30

# Body-wall muscles 07–23 (locomotion) — copied verbatim from worm-sim,
# typos and all (e.g. mLeft contains "MDL21" where the right side has "MDL21"
# duplicated; that's how the GoPiGo reference encoded it and changing it
# alters behavior).
M_LEFT = [
    "MDL07","MDL08","MDL09","MDL10","MDL11","MDL12","MDL13","MDL14","MDL15",
    "MDL16","MDL17","MDL18","MDL19","MDL20","MDL21","MDL22","MDL23",
    "MVL07","MVL08","MVL09","MVL10","MVL11","MVL12","MVL13","MVL14","MVL15",
    "MVL16","MVL17","MVL18","MVL19","MVL20","MVL21","MVL22","MVL23",
]
M_RIGHT = [
    "MDR07","MDR08","MDR09","MDR10","MDR11","MDR12","MDR13","MDR14","MDR15",
    "MDR16","MDR17","MDR18","MDR19","MDR20","MDL21","MDR22","MDR23",
    "MVR07","MVR08","MVR09","MVR10","MVR11","MVR12","MVR13","MVR14","MVR15",
    "MVR16","MVR17","MVR18","MVR19","MVR20","MVL21","MVR22","MVR23",
]
MUSCLE_LIST = M_LEFT + M_RIGHT

HUNGER_NEURONS = ("RIML", "RIMR", "RICL", "RICR")
NOSE_TOUCH_NEURONS = ("FLPR", "FLPL", "ASHL", "ASHR",
                      "IL1VL", "IL1VR", "OLQDL", "OLQDR", "OLQVR", "OLQVL")
FOOD_SENSE_NEURONS = ("ADFL", "ADFR", "ASGR", "ASGL",
                      "ASIL", "ASIR", "ASJR", "ASJL")

# Chemosensory / "taste" amphid neurons.
CHEMOSENSORY_NEURONS = frozenset({
    "ASEL","ASER",
    "AWAL","AWAR","AWBL","AWBR","AWCL","AWCR",
    "ADLL","ADLR","ADFL","ADFR",
    "ASHL","ASHR","ASIL","ASIR","ASJL","ASJR","ASKL","ASKR","ASGL","ASGR",
    "AFDL","AFDR",
})

# All sensory neurons (chemosensory + mechanosensory + others).
SENSORY_NEURONS = CHEMOSENSORY_NEURONS | frozenset({
    "PHAL","PHAR","PHBL","PHBR",
    "ALML","ALMR","AVM","PLML","PLMR","PVM",
    "PVDL","PVDR","FLPL","FLPR",
    "ADEL","ADER","PDEL","PDER",
    "CEPDL","CEPDR","CEPVL","CEPVR",
    "IL1DL","IL1DR","IL1L","IL1R","IL1VL","IL1VR",
    "IL2DL","IL2DR","IL2L","IL2R","IL2VL","IL2VR",
    "OLLL","OLLR",
    "OLQDL","OLQDR","OLQVL","OLQVR",
    "BAGL","BAGR","URXL","URXR",
    "AQR","PQR","BDUL","BDUR",
})

# Ventral cord + head motor neurons.
MOTOR_NEURONS = frozenset({
    "VA1","VA2","VA3","VA4","VA5","VA6","VA7","VA8","VA9","VA10","VA11","VA12",
    "VB1","VB2","VB3","VB4","VB5","VB6","VB7","VB8","VB9","VB10","VB11",
    "VC1","VC2","VC3","VC4","VC5","VC6",
    "VD1","VD2","VD3","VD4","VD5","VD6","VD7","VD8","VD9","VD10","VD11","VD12","VD13",
    "DA1","DA2","DA3","DA4","DA5","DA6","DA7","DA8","DA9",
    "DB1","DB2","DB3","DB4","DB5","DB6","DB7",
    "DD1","DD2","DD3","DD4","DD5","DD6",
    "AS1","AS2","AS3","AS4","AS5","AS6","AS7","AS8","AS9","AS10","AS11",
    "RMEL","RMER","RMED","RMEV",
    "RMDDL","RMDDR","RMDL","RMDR","RMDVL","RMDVR",
    "SMDDL","SMDDR","SMDVL","SMDVR",
    "SMBDL","SMBDR","SMBVL","SMBVR",
    "URADL","URADR","URAVL","URAVR",
    "DVB","AVL","HSNL","HSNR","PDA","PDB",
})


class Connectome:
    def __init__(
        self,
        weights: dict[str, dict[str, int]] | None = None,
        rng: random.Random | None = None,
    ):
        if weights is None:
            with open(WEIGHTS_PATH) as f:
                weights = json.load(f)
        self.weights = weights
        self.rng = rng if rng is not None else random.Random()

        neurons: set[str] = set(weights.keys())
        for targets in weights.values():
            neurons.update(targets.keys())
        # Make sure every muscle exists as a post-synaptic bucket even if it
        # never receives a synapse in the weights table.
        neurons.update(MUSCLE_LIST)
        self.neurons: list[str] = sorted(neurons)
        self.psyn: dict[str, list[float]] = {n: [0.0, 0.0] for n in self.neurons}

        self.this_state = 0
        self.next_state = 1
        self.accum_left = 0.0
        self.accum_right = 0.0
        # Per-muscle activations as captured during the most recent
        # motorcontrol() call. Keyed by muscle name (MDL07..MVR23). Values
        # decay slowly between brain ticks so the body has something to read
        # at 60 Hz even though the brain only ticks at 2 Hz.
        self.muscle_activations: dict[str, float] = {m: 0.0 for m in MUSCLE_LIST}

        self._left_set = set(M_LEFT)
        self._right_set = set(M_RIGHT)

        # --- lifelike plasticity state (inert unless enable_plasticity()) ---
        # self.weights stays the PRISTINE genome — rollovers and elites read
        # it — while learned changes live in _delta (effective weight =
        # genome + delta). Deltas die with the worm: Darwinian, not
        # Lamarckian. _trace holds per-edge eligibility (recent co-firing);
        # _fired accumulates firings across the several run() calls one
        # brain tick makes, consumed by plasticity_step().
        self._plasticity_on = False
        self._plast: dict[str, float] = {}
        self._delta: dict[str, dict[str, float]] = {}
        self._trace: dict[tuple[str, str], float] = {}
        # Memoised plasticity_stats(). The stats only change when
        # plasticity_step mutates _delta/_trace (2 Hz), but the focus
        # broadcast reads them at 60 Hz — on a saturated worm (~3k edges)
        # that was a ~3k-iteration loop 60×/s for a value that moves twice a
        # second. Any code that mutates _delta/_trace directly (not via
        # plasticity_step) must reset this to None.
        self._stats_cache: dict | None = None
        self._fired: set[str] = set()

    def enable_plasticity(self, params: dict[str, float]) -> None:
        self._plasticity_on = True
        self._plast = params

    # --- core ---
    def dendrite_accumulate(self, pre: str, scale: float = 1.0) -> None:
        targets = self.weights.get(pre)
        if not targets:
            return
        ns = self.next_state
        dpre = self._delta.get(pre)
        if dpre is None:
            for post, w in targets.items():
                self.psyn[post][ns] += w * scale
        else:
            for post, w in targets.items():
                self.psyn[post][ns] += (w + dpre.get(post, 0.0)) * scale

    def fire_neuron(self, neuron: str) -> None:
        if neuron == "MVULVA":
            return
        if self._plasticity_on:
            self._fired.add(neuron)
        self.dendrite_accumulate(neuron)
        self.psyn[neuron][self.next_state] = 0.0

    def motorcontrol(self) -> None:
        self.accum_left = 0.0
        self.accum_right = 0.0
        ns = self.next_state
        for m in MUSCLE_LIST:
            v = self.psyn[m][ns]
            self.muscle_activations[m] = v
            if m in self._left_set:
                self.accum_left += v
            elif m in self._right_set:
                self.accum_right += v
            self.psyn[m][ns] = 0.0

    def run(self) -> None:
        ts = self.this_state
        for n in self.neurons:
            if n[:3] in MUSCLE_PREFIXES:
                continue
            if self.psyn[n][ts] > FIRE_THRESHOLD:
                self.fire_neuron(n)
        self.motorcontrol()
        ns = self.next_state
        for n in self.neurons:
            self.psyn[n][ts] = self.psyn[n][ns]
        self.this_state, self.next_state = self.next_state, self.this_state

    # --- convenience ---
    def rand_excite(self, k: int = 40) -> None:
        keys = list(self.weights.keys())
        for _ in range(k):
            self.dendrite_accumulate(self.rng.choice(keys))

    def stimulate(self, neuron_names) -> None:
        for n in neuron_names:
            self.dendrite_accumulate(n)
        self.run()

    def stimulate_weighted(self, activations: dict) -> None:
        """Fire each named neuron with strength proportional to its
        activation in [0, 1]. Used by the PCA-driven chemosensation in
        v6.1+ so the brain actually consumes the rich sensory signal
        (previously it only saw the binary food_sense flag)."""
        for name, w in activations.items():
            if w > 0:
                self.dendrite_accumulate(name, scale=float(w))
        self.run()

    def tick(self, *, hunger: bool, nose_touch: bool, food_sense: bool) -> None:
        if hunger:
            self.stimulate(HUNGER_NEURONS)
        if nose_touch:
            self.stimulate(NOSE_TOUCH_NEURONS)
        if food_sense:
            self.stimulate(FOOD_SENSE_NEURONS)

    def activity(self) -> dict[str, float]:
        ts = self.this_state
        return {n: self.psyn[n][ts] for n in self.neurons}

    # --- lifelike plasticity (called once per brain tick by World) ---
    def plasticity_step(self, reward: float) -> None:
        """Three-factor learning rule. (1) Eligibility: edges whose pre AND
        post fired during this brain tick get their trace bumped — Hebbian
        co-activity marks candidate synapses. (2) Reward: eating consolidates
        every currently-traced edge into _delta, scaled by eta — synapses
        active on the path to food strengthen (positive weights up,
        inhibitory weights deepen: the delta follows the trace regardless of
        sign, reinforcing the circuit as wired). (3) Decay: traces fade
        (trace_decay) and deltas relax toward the genome (baseline_pull), so
        unreinforced learning is forgotten. All arithmetic is deterministic."""
        if not self._plasticity_on:
            return
        p = self._plast
        fired = self._fired
        if fired:
            trace = self._trace
            # sorted: set iteration order varies with PYTHONHASHSEED across
            # processes; per-key arithmetic is order-safe, but insertion order
            # feeds the plasticity_stats() L1 summation order and the
            # checkpoint's JSON byte order — sorting keeps those reproducible
            # across restarts.
            for pre in sorted(fired):
                targets = self.weights.get(pre)
                if not targets:
                    continue
                for post in targets:
                    if post in fired:
                        trace[(pre, post)] = trace.get((pre, post), 0.0) + 1.0
            self._fired = set()

        if reward > 0.0 and self._trace:
            from sim.lifelike import DELTA_CAP
            gain = p["eta"] * reward
            delta = self._delta
            for (pre, post), e in self._trace.items():
                # Reinforce the circuit AS WIRED: excitatory synapses
                # strengthen (+), inhibitory synapses deepen (−). An unsigned
                # increment would erode inhibition every meal and could drive
                # an inhibitory weight across zero into excitation.
                sign = 1.0 if self.weights.get(pre, {}).get(post, 0.0) >= 0 else -1.0
                dpre = delta.setdefault(pre, {})
                d = dpre.get(post, 0.0) + gain * e * sign
                if d > DELTA_CAP:
                    d = DELTA_CAP
                elif d < -DELTA_CAP:
                    d = -DELTA_CAP
                dpre[post] = d

        # Decay traces and relax deltas toward the genome; prune dust so the
        # sparse dicts stay small over a long life.
        from sim.lifelike import PRUNE_EPS
        td = p["trace_decay"]
        self._trace = {k: v * td for k, v in self._trace.items() if v * td > PRUNE_EPS}
        pull = 1.0 - p["baseline_pull"]
        if self._delta:
            new_delta: dict[str, dict[str, float]] = {}
            for pre, posts in self._delta.items():
                kept = {post: d * pull for post, d in posts.items()
                        if abs(d * pull) > PRUNE_EPS}
                if kept:
                    new_delta[pre] = kept
            self._delta = new_delta
        self._stats_cache = None

    def delta_norm(self) -> float:
        """L1 size of current learned changes — kept as the historical name;
        the one implementation lives in plasticity_stats()."""
        return self.plasticity_stats()["l1"]

    def plasticity_stats(self) -> dict:
        """Rollup of the learned layer for post-mortems and diagnostics.
        The L1 alone proved ambiguous in the field: three worms in one
        flask all reported the identical L1 (29620.0 — 2,962 edges pinned at
        DELTA_CAP) and one thrived while two starved. The capped-edge count is
        what separates 'learned a lot' from 'every traced synapse slammed into
        the cap and the rule can no longer steer'."""
        if self._stats_cache is None:
            from sim.lifelike import DELTA_CAP
            near_cap = DELTA_CAP - 1e-9
            edges = capped = 0
            l1 = 0.0
            for posts in self._delta.values():
                for d in posts.values():
                    a = abs(d)
                    edges += 1
                    l1 += a
                    if a >= near_cap:
                        capped += 1
            self._stats_cache = {"edges": edges, "capped": capped,
                                 "l1": round(l1, 2), "traces": len(self._trace)}
        return dict(self._stats_cache)
