"""World: ticks the connectome (slow) and the body (fast), and tracks food.

Deterministic in v6: the world is driven by a tick counter, not wallclock,
so given the same (weights, seed, debug-input-schedule) two runs produce
identical eaten-word sequences.

The body ticks every call to tick(); the brain fires every BRAIN_TICK_PERIOD
body ticks. Stimulus linger is measured in ticks, not seconds.

v6.1+: chemosensation is driven by a 12-dim PCA over the full Hamlet
embedding corpus, not the 8-dim NRC emotion lexicon. Eaten words leave
a decaying residual in the chemosensory neurons for ~10 brain ticks.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

from sim.connectome import Connectome
from sim.muscle_body import MuscleBody
from sim.worm import WormBody
from sim.text_scroller import TextScroller
from corpus.hamlet import get_sentences_with_flags
from sim.chemosensory_mapping import (
    compute_pca_activation, PC_NEURON_PAIRS,
)
from server import embedding


# World extents in the same arbitrary "pixel" units as worm-sim.
WORLD_W = 1600.0
WORLD_H = 1000.0

# Body ticks 60 Hz; brain ticks every 30 body ticks (2 Hz).
BODY_TICK_HZ = 60
BRAIN_TICK_PERIOD = 30
BODY_DT = 1.0 / BODY_TICK_HZ

# Food parameters (worm-sim values).
FOOD_SENSE_RADIUS = 200.0  # Smell detection distance (extended for visualization)
FOOD_EAT_RADIUS = 20.0
# How long a sensory stimulus lingers after the trigger fires (in body ticks).
STIM_LINGER_TICKS = int(2.0 * BODY_TICK_HZ)  # 2 seconds

# v7 chemosensory drive: a LEARNED, context-aware embedding (server/embedding.py)
# replaces v6's frozen 12-dim UMAP. For each in-range word we compute a 12-dim
# vector from (a) the word's frozen 512-dim nomic vector and (b) the worm's
# last-N eaten words, then map dim i → its amphid neuron pair via PC_NEURON_PAIRS.
# Dims 0..10 are the embedding net; dim 11 is the POS syntax net. There is NO
# decaying residual anymore — the worm's memory IS the eaten-word history fed
# into the embedding, so the same word tastes different depending on context.
HISTORY_LEN = embedding.HISTORY  # number of eaten words used as chemosensory context


@dataclass
class Food:
    x: float
    y: float
    word: str = ""
    line_id: int = -1
    word_idx: int = -1
    edible: bool = True  # False = inert set-dressing (no eat, no smell)


@dataclass
class World:
    """Deterministic per-worm world. Pass `seed` to control all randomness
    (initial brain excitation, and any future per-worm randomness). Pass
    `weights` to give this worm its own copy of the connectome weights so
    different worms can drift apart in v7+ without affecting each other.

    body_kind=`ik` (default) is worm-sim's kinematic IK chain — the head
    follows a steered "target" point, the body trails via spring relaxation.
    `muscle` integrates the midline directly from per-segment dorsal-ventral
    muscle differentials read from the connectome (kept for the eventual
    stretch-feedback fix; not used by v6 by default)."""
    seed: int = 0
    weights: dict | None = None
    # v7: this worm's own perturbation of the shared embedding genome. None →
    # use the process-wide default model (used by tests / legacy paths).
    embedding_genome: list | None = None
    body_kind: str = "ik"
    brain: Connectome = field(init=False)
    worm: WormBody | MuscleBody = field(init=False)
    food: list[Food] = field(default_factory=list)
    text_scroller: TextScroller = field(init=False)
    # Smell tracking: {word_key: emotion_vector, ...}
    sensed_smells: dict = field(default_factory=dict)

    # State flags, mirroring worm-sim.
    stim_hunger: bool = True
    stim_nose_touch: bool = False
    stim_food_sense: bool = False
    _stim_linger_until: int = 0
    tick_count: int = 0
    paused: bool = False
    # Words eaten since last drain (line_id, word_idx, word) — consumed by
    # the orchestrator to append to disk and broadcast.
    _eaten_buffer: list[tuple[int, int, str]] = field(default_factory=list)
    # v7 memory: the last HISTORY_LEN eaten words, most-recent-first. Fed to the
    # learned embedding as chemosensory context (replaces v6's decaying residual).
    _recent_eaten: list = field(default_factory=list)
    # Brain tick counter (kept for pacing; no longer drives any decay math).
    _brain_tick_count: int = 0

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.brain = Connectome(weights=self.weights, rng=self.rng)
        # v7 chemosensory embedding model: this worm's own perturbation of the
        # shared genome if given, else the process-wide default (random-init).
        if self.embedding_genome is not None:
            import numpy as _np
            self.embedding_model = embedding.EmbeddingModel(
                embedding.EmbeddingParams.from_flat(_np.asarray(self.embedding_genome, dtype=float))
            )
        else:
            self.embedding_model = embedding.get_model()
        if self.body_kind == "muscle":
            self.worm = MuscleBody(
                origin_x=WORLD_W / 2, origin_y=WORLD_H / 2,
                body_length=800.0,
            )
        elif self.body_kind == "ik":
            self.worm = WormBody(
                n_segments=200, segment_size=4.0,
                origin_x=WORLD_W / 2, origin_y=WORLD_H / 2,
                rng=self.rng,
            )
        else:
            raise ValueError(f"unknown body_kind: {self.body_kind!r}")
        self.brain.rand_excite()
        # When generational evolution is enabled, scroll the full play once
        # per generation; otherwise loop the opening passage forever (legacy
        # v6 behavior).
        passage = os.environ.get("WORMLET_PASSAGE", "opening")
        loop = os.environ.get("WORMLET_GENERATIONS_ENABLED", "0") != "1"
        sentences, edible_flags = get_sentences_with_flags(passage)
        self.text_scroller = TextScroller(sentences, loop=loop, edible_flags=edible_flags)

    def add_food(self, x: float, y: float) -> None:
        self.food.append(Food(x, y))

    def clear_food(self) -> None:
        self.food.clear()

    def _compute_smells(self) -> None:
        """For each in-range word, compute the 12-d PCA-driven chemosensory
        signal it produces in the worm's nose. Distance fades intensity;
        direction biases L vs R within each PC's neuron pair."""
        wx = self.worm.target_x
        wy = self.worm.target_y
        self.sensed_smells.clear()

        for f in self.food:
            if not f.word or not f.edible:
                continue
            chemo = self.embedding_model.embed(f.word, self._recent_eaten)
            if chemo is None:
                # Word isn't in the embedded corpus (e.g. punctuation, OOV).
                # Skip — no chemosensory signal.
                continue
            pca = [float(v) for v in chemo]  # 12-dim in [0,1] (11 emb + 1 POS)
            dx = f.x - wx
            dy = f.y - wy
            d = math.hypot(dx, dy)
            if d > FOOD_SENSE_RADIUS:
                continue

            key = f"{f.line_id}_{f.word_idx}"
            distance_factor = 1.0 - (d / FOOD_SENSE_RADIUS)

            # Direction-of-arrival relative to facing. angle_diff > 0 → food
            # is to the worm's left; < 0 → right.
            food_angle = math.atan2(dy, dx)
            angle_diff = (food_angle - self.worm.facing_dir + math.pi) % (2 * math.pi) - math.pi
            direction_factor = 0.5 + 0.5 * math.sin(angle_diff)  # [0, 1]

            neuron_activation = compute_pca_activation(
                pca, direction_factor=direction_factor,
                intensity=distance_factor,
            )

            self.sensed_smells[key] = {
                "word": f.word,
                "x": f.x,
                "y": f.y,
                "distance": d,
                "pca": pca,
                "direction_factor": direction_factor,
                "distance_factor": distance_factor,
                "neurons": neuron_activation,
            }

    def _check_food(self) -> None:
        wx = self.worm.target_x
        wy = self.worm.target_y
        i = 0
        while i < len(self.food):
            f = self.food[i]
            if not f.edible:
                i += 1
                continue
            d = math.hypot(wx - f.x, wy - f.y)
            if d <= FOOD_SENSE_RADIUS:
                self.stim_food_sense = True
                self._stim_linger_until = self.tick_count + STIM_LINGER_TICKS
                if d <= FOOD_EAT_RADIUS:
                    if f.word:  # it's a Hamlet word
                        self.text_scroller.mark_eaten(f.line_id, f.word_idx)
                        self._eaten_buffer.append((f.line_id, f.word_idx, f.word))
                        # Update the worm's chemosensory memory (most-recent-first),
                        # capped at HISTORY_LEN. This IS the worm's residual now.
                        self._recent_eaten.insert(0, f.word)
                        del self._recent_eaten[HISTORY_LEN:]
                    self.food.pop(i)
                    continue
            i += 1

    def _check_walls(self) -> None:
        if self.worm.target_x < 0:
            self.worm.target_x = 0
            self.stim_nose_touch = True
            self._stim_linger_until = self.tick_count + STIM_LINGER_TICKS
        elif self.worm.target_x > WORLD_W:
            self.worm.target_x = WORLD_W
            self.stim_nose_touch = True
            self._stim_linger_until = self.tick_count + STIM_LINGER_TICKS
        if self.worm.target_y < 0:
            self.worm.target_y = 0
            self.stim_nose_touch = True
            self._stim_linger_until = self.tick_count + STIM_LINGER_TICKS
        elif self.worm.target_y > WORLD_H:
            self.worm.target_y = WORLD_H
            self.stim_nose_touch = True
            self._stim_linger_until = self.tick_count + STIM_LINGER_TICKS

    def tick(self) -> None:
        """Advance one body tick. Deterministic — no wallclock reads."""
        if self.paused:
            return

        # Text scroller tick (before food check). Uses fixed dt.
        self.text_scroller.step(BODY_DT)

        # Rebuild food from alive scrolling words.
        self.food = [
            Food(x=w.x, y=w.y, word=w.text, line_id=w.line_id, word_idx=w.word_idx,
                 edible=w.edible)
            for w in self.text_scroller.alive_words()
        ]

        # Brain at 2 Hz (every BRAIN_TICK_PERIOD body ticks).
        if self.tick_count % BRAIN_TICK_PERIOD == 0:
            # v7: chemosensation is expensive now (a learned embedding forward
            # pass per in-range word), and the brain only reads it here at 2 Hz.
            # So compute smells ONLY on brain ticks instead of every body tick —
            # 30× fewer embedding passes, which is the dominant per-tick cost.
            self._compute_smells()
            self.brain.tick(
                hunger=self.stim_hunger,
                nose_touch=self.stim_nose_touch,
                food_sense=self.stim_food_sense,
            )
            # Push the rich per-neuron chemosensory signal into the brain.
            chemo = self._chemo_pulse()
            if chemo:
                self.brain.stimulate_weighted(chemo)
            self._brain_tick_count += 1
            if isinstance(self.worm, WormBody):
                self.worm.consume_motor(self.brain.accum_left, self.brain.accum_right)

        # Body every tick. MuscleBody re-reads per-muscle activations with
        # smoothing; the IK body is driven only at brain ticks via
        # consume_motor() above.
        if isinstance(self.worm, MuscleBody):
            self.worm.consume(self.brain)
        self.worm.step()
        self._check_food()
        self._check_walls()

        # Decay transient stimuli back to "just hungry" after the linger window.
        if self.tick_count >= self._stim_linger_until:
            self.stim_hunger = True
            self.stim_nose_touch = False
            self.stim_food_sense = False

        self.tick_count += 1

    def drain_eaten_words(self) -> list[tuple[int, int, str]]:
        """Return and clear the buffer of words eaten since the last drain."""
        out = self._eaten_buffer
        self._eaten_buffer = []
        return out

    def _chemo_pulse(self) -> dict[str, float]:
        """Per-neuron chemosensory activation from in-range smells (v7: no
        residual term — memory lives in the context-aware embedding of each
        in-range word). Each smell already carries direction-aware L/R splits."""
        out: dict[str, float] = {}
        for smell in self.sensed_smells.values():
            for n, v in smell.get("neurons", {}).items():
                if v > 0:
                    out[n] = out.get(n, 0.0) + v
        # Saturate at 1.0 per neuron.
        for k in list(out.keys()):
            if out[k] > 1.0:
                out[k] = 1.0
        return out

    def snapshot(self) -> dict:
        """Serializable state for the viewer."""
        midline = self.worm.midline()

        # Sparse neural activity: only neurons with charge above the visible
        # threshold to keep JSON payload light.
        VIS_THRESHOLD = 1.0
        raw = self.brain.activity()
        neurons_active = {
            n: round(v, 1) for n, v in raw.items() if v > VIS_THRESHOLD
        }

        return {
            "world": {"w": WORLD_W, "h": WORLD_H},
            "body_kind": self.body_kind,
            "midline": [[round(x, 2), round(y, 2)] for x, y in midline],
            "head": [round(self.worm.target_x, 2), round(self.worm.target_y, 2)],
            "facing": round(self.worm.facing_dir, 4),
            "speed": round(self.worm.speed, 4),
            "food": [
                {
                    "x": round(f.x, 2),
                    "y": round(f.y, 2),
                    "word": f.word,
                    "line_id": f.line_id,
                    "word_idx": f.word_idx,
                }
                for f in self.food
            ],
            "smells": [
                {
                    "word": smell["word"],
                    "x": round(smell["x"], 2),
                    "y": round(smell["y"], 2),
                    "distance": round(smell["distance"], 2),
                    # v7: 12-dim learned chemosensory vector (11 embedding + 1 POS)
                    "pca": [round(v, 4) for v in smell["pca"]],
                    "neurons": smell["neurons"],
                }
                for smell in self.sensed_smells.values()
            ],
            "recent_eaten": list(self._recent_eaten),
            "motor": {"L": round(self.brain.accum_left, 2),
                      "R": round(self.brain.accum_right, 2)},
            "stim": {"hunger": self.stim_hunger,
                     "nose_touch": self.stim_nose_touch,
                     "food_sense": self.stim_food_sense},
            "paused": self.paused,
            "neurons": neurons_active,
        }
