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

    # --- core ---
    def dendrite_accumulate(self, pre: str, scale: float = 1.0) -> None:
        targets = self.weights.get(pre)
        if not targets:
            return
        ns = self.next_state
        for post, w in targets.items():
            self.psyn[post][ns] += w * scale

    def fire_neuron(self, neuron: str) -> None:
        if neuron == "MVULVA":
            return
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
