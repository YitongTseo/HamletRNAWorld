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
from corpus.hamlet import get_sentences
from sim.chemosensory_mapping import (
    compute_pca_activation, PC_NEURON_PAIRS,
)


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

# Residual decay for the eaten-word chemosensory afterglow. Measured in
# BRAIN ticks (not body ticks). With τ=12 brain ticks @ 2 Hz, a word's
# residual half-life is ~4.2 sec and falls below the 1% eviction floor
# at ~55 brain ticks ≈ 28 sec — within the C. elegans short-term
# chemosensory memory window.
RESIDUAL_TAU_BRAIN_TICKS = 12.0
RESIDUAL_EVICT_THRESHOLD = 0.01

# Chemosensory drive source. Either PCA or UMAP over the full Hamlet
# embedding corpus produces a 12-d vector per word; PC_NEURON_PAIRS maps
# dim i → its amphid neuron pair regardless of which produced it.
# Set WORMLET_EMBEDDING=pca to revert to the linear projection.
_V6_ROOT = Path(__file__).resolve().parent.parent
_CORPUS_PCA_PATH = _V6_ROOT / "cache" / "corpus_pca.json"
_CORPUS_UMAP_PATH = _V6_ROOT / "cache" / "corpus_umap.json"
EMBEDDING_MODE = os.environ.get("WORMLET_EMBEDDING", "umap").lower()
_CORPUS_PCA_CACHE: dict | None = None


def _load_corpus_pca() -> dict[str, list[float]]:
    """Return {lowercased_word: [d0..d11]} for chemosensory drive.

    Source is selected by the WORMLET_EMBEDDING env var:
      - 'umap' (default): cache/corpus_umap.json, key umap12_sparse/umap12
      - 'pca'           : cache/corpus_pca.json,  key pca12_sparse/pca12

    Function name retained for backward compat; the cache shape is identical
    either way. Loads once per process and shared across all worms."""
    global _CORPUS_PCA_CACHE
    if _CORPUS_PCA_CACHE is not None:
        return _CORPUS_PCA_CACHE

    if EMBEDDING_MODE == "umap":
        path = _CORPUS_UMAP_PATH
        sparse_key, raw_key = "umap12_sparse", "umap12"
    else:
        path = _CORPUS_PCA_PATH
        sparse_key, raw_key = "pca12_sparse", "pca12"

    if not path.exists():
        # Fall back to empty map; chemosensation goes silent. Smoke tests
        # run without the artifact and shouldn't crash.
        _CORPUS_PCA_CACHE = {}
    else:
        data = json.loads(path.read_text())
        words = data["words"]
        vecs = data.get(sparse_key) or data[raw_key]
        _CORPUS_PCA_CACHE = {w: v for w, v in zip(words, vecs)}
    return _CORPUS_PCA_CACHE


def _word_pca(word: str) -> list[float] | None:
    """Lookup PCA for a single word, case-insensitive + apostrophe-tolerant."""
    table = _load_corpus_pca()
    if not table:
        return None
    key = word.lower().strip("'").strip()
    return table.get(key)


@dataclass
class Food:
    x: float
    y: float
    word: str = ""
    line_id: int = -1
    word_idx: int = -1


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
    # Decaying chemosensory afterglow: list of (pca12, word, eaten_at_brain_tick).
    # Unbounded; entries are evicted once their contribution drops below
    # RESIDUAL_EVICT_THRESHOLD. See compute_residual_chemo().
    _eaten_history: list = field(default_factory=list)
    # Track brain ticks so the residual decay math is clock-independent.
    _brain_tick_count: int = 0
    # Last computed residual + per-word decay info, for the viewer.
    _last_residual_pca: list = field(default_factory=lambda: [0.0] * 12)
    _last_residual_words: list = field(default_factory=list)

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.brain = Connectome(weights=self.weights, rng=self.rng)
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
        self.text_scroller = TextScroller(get_sentences(passage), loop=loop)

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
            if not f.word:
                continue
            pca = _word_pca(f.word)
            if pca is None:
                # Word isn't in the embedded corpus (e.g. punctuation, OOV).
                # Skip — no chemosensory signal.
                continue
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
            d = math.hypot(wx - f.x, wy - f.y)
            if d <= FOOD_SENSE_RADIUS:
                self.stim_food_sense = True
                self._stim_linger_until = self.tick_count + STIM_LINGER_TICKS
                if d <= FOOD_EAT_RADIUS:
                    if f.word:  # it's a Hamlet word
                        self.text_scroller.mark_eaten(f.line_id, f.word_idx)
                        self._eaten_buffer.append((f.line_id, f.word_idx, f.word))
                        pca = _word_pca(f.word)
                        if pca is not None:
                            self._eaten_history.append(
                                (pca, f.word, self._brain_tick_count)
                            )
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
            Food(x=w.x, y=w.y, word=w.text, line_id=w.line_id, word_idx=w.word_idx)
            for w in self.text_scroller.alive_words()
        ]

        # Brain at 2 Hz (every BRAIN_TICK_PERIOD body ticks).
        if self.tick_count % BRAIN_TICK_PERIOD == 0:
            self.brain.tick(
                hunger=self.stim_hunger,
                nose_touch=self.stim_nose_touch,
                food_sense=self.stim_food_sense,
            )
            # Weighted chemosensory pulse: combine in-range smells with the
            # decaying residual from previously eaten words, then push the
            # rich PC-driven signal into the chemosensory neurons.
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
        self._compute_smells()
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

    def _residual_pca(self) -> tuple[list[float], list[tuple[str, float]]]:
        """Compute the current decaying residual PCA vector from previously
        eaten words, evicting entries below threshold. Returns (residual_vec,
        per_word_decay) where per_word_decay is a small list for the viewer."""
        if not self._eaten_history:
            return [0.0] * 12, []
        now = self._brain_tick_count
        kept = []
        residual = [0.0] * 12
        per_word: list[tuple[str, float]] = []
        for pca, word, eaten_at in self._eaten_history:
            age = now - eaten_at
            decay = math.exp(-age / RESIDUAL_TAU_BRAIN_TICKS)
            if decay < RESIDUAL_EVICT_THRESHOLD:
                continue   # too old, drop it
            kept.append((pca, word, eaten_at))
            for i, v in enumerate(pca):
                if i >= 12:
                    break
                residual[i] += v * decay
            per_word.append((word, round(decay, 3)))
        self._eaten_history = kept
        # The residual can grow large if many words pile up at low decay;
        # saturate per-dim at 1.0 so we don't overdrive the neurons.
        residual = [min(1.0, v) for v in residual]
        return residual, per_word

    def _chemo_pulse(self) -> dict[str, float]:
        """Combine in-range smells (with full directional asymmetry) and the
        eaten-word residual (no direction; same to L and R) into one
        per-neuron activation dict to push into the brain."""
        out: dict[str, float] = {}
        # In-range smells: already have direction-aware L/R splits.
        for smell in self.sensed_smells.values():
            for n, v in smell.get("neurons", {}).items():
                if v > 0:
                    out[n] = out.get(n, 0.0) + v
        # Residual: contributes equally to L and R (food is inside the worm,
        # there's no direction of arrival anymore).
        residual, per_word = self._residual_pca()
        for i, v in enumerate(residual):
            if v <= 0 or i >= len(PC_NEURON_PAIRS):
                continue
            L, R = PC_NEURON_PAIRS[i]
            out[L] = out.get(L, 0.0) + 0.5 * v
            out[R] = out.get(R, 0.0) + 0.5 * v
        # Saturate at 1.0 per neuron.
        for k in list(out.keys()):
            if out[k] > 1.0:
                out[k] = 1.0
        # Stash a viewer-facing copy of the residual for the snapshot.
        self._last_residual_pca = residual
        self._last_residual_words = per_word
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
                    "emotions": smell["emotions"],
                    "weighted_emotions": smell["weighted_emotions"],
                    "neurons": smell["neurons"],
                }
                for smell in self.sensed_smells.values()
            ],
            "motor": {"L": round(self.brain.accum_left, 2),
                      "R": round(self.brain.accum_right, 2)},
            "stim": {"hunger": self.stim_hunger,
                     "nose_touch": self.stim_nose_touch,
                     "food_sense": self.stim_food_sense},
            "paused": self.paused,
            "neurons": neurons_active,
        }
