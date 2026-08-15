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
from sim import lifelike
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
    # v7.1: the SHARED, per-flask embedding model. All worms in a flask pass the
    # SAME EmbeddingModel instance here, so its precomputed E-table is built once
    # per generation and reused across the whole flask. None → the process-wide
    # default model (used by tests / legacy / single-worm paths).
    embedding_model: "embedding.EmbeddingModel | None" = None
    # Which text this world scrolls (WORMLET_FLASK_TEXTS, per flask). None →
    # legacy env-passage hamlet, so every pre-multi-corpus caller is
    # unchanged. Non-hamlet corpora always scroll their full text.
    corpus: str | None = None
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
    # Lifelike mode (sim/lifelike.py). Flags are read once at init so a
    # worm's behaviour can't change mid-life if the env mutates.
    satiety: float = lifelike.SATIETY_START
    dead: bool = False
    _reward_accum: float = 0.0
    # Motion tracking for the spin-from-birth watch (BEHAVIOUR-LOG.md
    # 2026-08-15): per brain tick (2 Hz), |heading change| and position ride
    # 60-second windows. A spinner shows high turns_min with drift_min under
    # a body length; observability only — no sim behaviour reads these.
    _turn_window: object = field(default_factory=lambda: __import__("collections").deque(maxlen=120))
    _pos_window: object = field(default_factory=lambda: __import__("collections").deque(maxlen=120))
    _last_facing: float | None = None
    # Habituation: per-neuron adapted baseline (EMA of recent pre-gain
    # chemosensory input). Response = input - baseline, so a static smell
    # fades and the worm moves on. Empty forever when the flag is off.
    _adapt: dict = field(default_factory=dict)
    # Post-mortem bookkeeping. tick_count keeps advancing for a corpse (the
    # scroller must exhaust for rollover), so the death tick and the gap since
    # the last meal exist only at the moment of death — capture them then.
    # Deterministic: sim ticks only, no wallclock (the server's death logger
    # adds the wall timestamp when it writes the record out).
    _last_meal_tick: int = -1
    death_record: dict | None = None

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        # Pop the _lifelike gene block BEFORE the dict reaches Connectome —
        # it's params, not a neuron. pop_params returns clipped values (or
        # defaults) whether or not the genome carries the block.
        self._plasticity_on = lifelike.plasticity_enabled()
        self._hunger_on = lifelike.hunger_enabled()
        self._habituation_on = lifelike.habituation_enabled()
        self.lifelike_params = lifelike.pop_params(self.weights)
        self.brain = Connectome(weights=self.weights, rng=self.rng)
        if self._plasticity_on:
            self.brain.enable_plasticity(self.lifelike_params)
        # v7.1 chemosensory embedding: the flask's SHARED model if given, else
        # the process-wide default (random-init). Shared across the flask so the
        # precomputed E-table is built once per generation, not per worm.
        if self.embedding_model is None:
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
        loop = os.environ.get("WORMLET_GENERATIONS_ENABLED", "0") != "1"
        if self.corpus is None or self.corpus == "hamlet":
            # Legacy/default path, byte-identical to stock v7: hamlet with
            # the env-selected passage.
            passage = os.environ.get("WORMLET_PASSAGE", "opening")
            sentences, edible_flags = get_sentences_with_flags(passage)
        else:
            from corpus import library
            sentences, edible_flags = library.get_sentences_with_flags(
                self.corpus, "full")
        # Pace non-hamlet corpora so every flask's generation lasts about as
        # long as hamlet act1 (~1500 lines at 4.5 s): the rollover is a JOINT
        # barrier across flasks, and with hunger on a short-corpus flask
        # would starve on an empty dish for hours waiting for a long one
        # (beowulf full = 4286 lines). Denser/sparser dishes change feeding
        # economics — recorded in the fidelity ledger. Hamlet keeps 4.5 s,
        # byte-identical to stock.
        if self.corpus is None or self.corpus == "hamlet":
            self.text_scroller = TextScroller(sentences, loop=loop,
                                              edible_flags=edible_flags)
        else:
            # Clamp ceiling 25 s: the daodejing is only 323 lines, so
            # equal-duration pacing means a sparse dish (~18 chars per
            # line every ~21 s — still ~60x the starvation line).
            interval = min(25.0, max(1.5, 4.5 * 1500.0 / max(1, len(sentences))))
            self.text_scroller = TextScroller(
                sentences, loop=loop, edible_flags=edible_flags,
                spawn_interval=interval,
                layout=library.LAYOUTS.get(self.corpus, "horizontal"))

    def add_food(self, x: float, y: float) -> None:
        self.food.append(Food(x, y))

    def clear_food(self) -> None:
        self.food.clear()

    def _compute_smells(self) -> None:
        """For each in-range word, compute the 12-d PCA-driven chemosensory
        signal it produces in the worm's nose. Distance fades intensity;
        direction biases L vs R within each PC's neuron pair.

        v7.1: distance-filter FIRST (cheap), then embed every in-range word in
        ONE batched forward pass sharing this worm's eaten history — instead of
        an embed() call per word that re-derived the history summary each time.
        Order-preserving (iterates self.food), so identical to the per-word path
        (verified to 1e-9) and bit-for-bit deterministic."""
        wx = self.worm.target_x
        wy = self.worm.target_y
        self.sensed_smells.clear()

        # 1) collect in-range edible words (order = self.food order).
        in_range = []  # (Food, dx, dy, d)
        for f in self.food:
            if not f.word or not f.edible:
                continue
            dx = f.x - wx
            dy = f.y - wy
            d = math.hypot(dx, dy)
            if d > FOOD_SENSE_RADIUS:
                continue
            in_range.append((f, dx, dy, d))
        if not in_range:
            return

        # 2) ONE batched embedding pass for the whole in-range set.
        words = [t[0].word for t in in_range]
        chemo_batch, valid = self.embedding_model.embed_batch(words, self._recent_eaten)

        # 3) per-word geometry → neuron activation.
        for row, ok, (f, dx, dy, d) in zip(chemo_batch, valid, in_range):
            if not ok:
                # Word isn't in the embedded corpus (e.g. punctuation, OOV).
                # Skip — no chemosensory signal (matches embed() -> None).
                continue
            pca = [float(v) for v in row]  # 12-dim in [0,1] (11 emb + 1 POS)

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
                        self._last_meal_tick = self.tick_count
                        # Lifelike: eating feeds the worm and/or rewards the
                        # brain. Reward scales with hunger — a meal found
                        # while starving consolidates harder (dopamine).
                        if self._hunger_on:
                            self._reward_accum += (
                                lifelike.REWARD_BASE + (1.0 - self.satiety))
                            self.satiety = min(1.0, self.satiety + lifelike.SATIETY_BITE)
                        elif self._plasticity_on:
                            self._reward_accum += 1.0
                        # Habituation: food is the salient event that
                        # restores a faded response (dishabituation).
                        if self._habituation_on:
                            self._dishabituate()
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

        # A starved worm is dead: no brain, no body, no eating. The scroller
        # keeps stepping above so the corpus still exhausts and the flask's
        # rollover fires on time regardless of casualties.
        if self.dead:
            self.tick_count += 1
            return

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
            # Spin watch: wrapped heading delta + position, 2 Hz.
            f = self.worm.facing_dir
            if self._last_facing is not None:
                self._turn_window.append(
                    abs((f - self._last_facing + math.pi) % (2 * math.pi) - math.pi))
            self._last_facing = f
            self._pos_window.append((self.worm.target_x, self.worm.target_y))
            # Lifelike: consolidate rewards accumulated since the last brain
            # tick (traces are fresh — the brain fired within trace memory of
            # the meal), then decay traces/deltas. No-op when plasticity off.
            if self._plasticity_on:
                self.brain.plasticity_step(self._reward_accum)
            # Drain regardless: with hunger on and plasticity off the accum
            # would otherwise grow unbounded for the whole life (harmless but
            # sloppy — audit finding).
            self._reward_accum = 0.0
            # Hunger: a starving worm roams — motor gain rises as satiety
            # falls (serotonin/dopamine roam-dwell switch, crudely).
            motor_gain = 1.0
            if self._hunger_on:
                motor_gain += self.lifelike_params["roam_gain"] * (1.0 - self.satiety)
            if isinstance(self.worm, WormBody):
                self.worm.consume_motor(self.brain.accum_left * motor_gain,
                                        self.brain.accum_right * motor_gain)

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

        # Hunger: metabolism runs every body tick; hitting zero is death.
        if self._hunger_on and not self.dead:
            self.satiety -= lifelike.SATIETY_DECAY_PER_TICK
            if self.satiety <= 0.0:
                self.satiety = 0.0
                self.dead = True
                self.death_record = {
                    "cause": "starvation",
                    "died_at_tick": self.tick_count,
                    "last_meal_tick": self._last_meal_tick,
                    # -1 last_meal_tick = never ate; the whole life was the fast.
                    "starved_ticks": (self.tick_count - self._last_meal_tick
                                      if self._last_meal_tick >= 0
                                      else self.tick_count),
                    "x": round(self.worm.target_x, 1),
                    "y": round(self.worm.target_y, 1),
                }

        self.tick_count += 1

    def lifelike_checkpoint(self) -> dict | None:
        """Serialisable within-life biology for the mid-generation checkpoint:
        satiety and the learned plasticity deltas/traces. Without this every
        restart was mild amnesia plus a free meal — the worm forgot what it
        had learned this life and respawned fed. None when both features are
        off, so stock checkpoints don't grow a key."""
        if not (self._hunger_on or self._plasticity_on or self._habituation_on):
            return None
        out: dict = {}
        if self._hunger_on:
            out["satiety"] = self.satiety
            out["dead"] = self.dead
            out["last_meal_tick"] = self._last_meal_tick
            if self.death_record is not None:
                # Carries the "logged" flag the server sets after writing
                # deaths.jsonl, so a restart doesn't re-log the same death.
                out["death_record"] = self.death_record
        if self._plasticity_on:
            out["delta"] = self.brain._delta
            out["trace"] = [[pre, post, v]
                            for (pre, post), v in self.brain._trace.items()]
        if self._habituation_on:
            # Without this a restart would dishabituate every nose for free.
            out["adapt"] = self._adapt
        return out

    def restore_lifelike(self, data: dict | None) -> None:
        """Inverse of lifelike_checkpoint. Tolerates None and missing keys
        (v1 checkpoints predate this), and ignores state for features that
        are currently off."""
        if not data:
            return
        if self._hunger_on and "satiety" in data:
            self.satiety = float(data["satiety"])
            self.dead = bool(data.get("dead", False))
            self._last_meal_tick = int(data.get("last_meal_tick", -1))
            self.death_record = data.get("death_record")
        if self._plasticity_on and "delta" in data:
            self.brain._delta = {pre: {post: float(v) for post, v in posts.items()}
                                 for pre, posts in data["delta"].items()}
            self.brain._trace = {(a, b): float(v)
                                 for a, b, v in data.get("trace", [])}
            self.brain._stats_cache = None  # direct _delta mutation
        if self._habituation_on and "adapt" in data:
            # Sorted for the same hash-seed-independent order the EMA keeps.
            self._adapt = {n: float(data["adapt"][n])
                           for n in sorted(data["adapt"])}

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
        # Hunger: starving worms smell harder — chemosensory gain rises as
        # satiety falls (real starved C. elegans show heightened chemotaxis).
        # Applied before saturation, so faint far-off words become salient to
        # a desperate worm but strong smells still cap at 1.0.
        gain = 1.0
        if self._hunger_on:
            gain += self.lifelike_params["starve_gain"] * (1.0 - self.satiety)
        if not self._habituation_on:
            # Stock path, verbatim: accumulating v*gain directly is NOT the
            # same floats as sum-then-multiply, and flags-off must stay
            # bit-identical to v7 — so the habituated path below is a
            # separate branch, not a refactor of this one.
            for smell in self.sensed_smells.values():
                for n, v in smell.get("neurons", {}).items():
                    if v > 0:
                        out[n] = out.get(n, 0.0) + v * gain
            # Saturate at 1.0 per neuron.
            for k in list(out.keys()):
                if out[k] > 1.0:
                    out[k] = 1.0
            return out
        # Habituation (called once per brain tick from tick(), so this is
        # also the adaptation clock). Baselines track the PRE-gain input:
        # adaptation is receptor-level, hunger gain is downstream — so a
        # worm growing desperate partially "un-adapts", which is the real
        # interplay (starvation re-sensitises chemotaxis).
        raw: dict[str, float] = {}
        for smell in self.sensed_smells.values():
            for n, v in smell.get("neurons", {}).items():
                if v > 0:
                    raw[n] = raw.get(n, 0.0) + v
        # EMA over the union: stimulated neurons adapt toward their input,
        # silent ones relax toward 0 (recovery) at the same rate. Sorted so
        # dict insertion order — and thus every downstream float sum — is
        # independent of PYTHONHASHSEED (determinism contract).
        rate = self.lifelike_params["adapt_rate"]
        adapt = self._adapt
        for n in sorted(set(adapt) | set(raw)):
            a = adapt.get(n, 0.0) * (1.0 - rate) + raw.get(n, 0.0) * rate
            if a > lifelike.PRUNE_EPS:
                adapt[n] = a
            else:
                adapt.pop(n, None)
        # Respond to input minus the (just-updated) baseline, floored at 0:
        # a fully adapted channel goes silent. Gain after subtraction, cap
        # at 1.0 as ever.
        for n, v in raw.items():
            e = (v - adapt.get(n, 0.0)) * gain
            if e > 0.0:
                out[n] = e if e < 1.0 else 1.0
        return out

    def _dishabituate(self) -> None:
        """Eating clears dishab_relief of every adapted baseline — the meal
        re-sensitises the nose (real dishabituation: a salient stimulus
        restores a habituated response). At relief=1 one bite is a full
        reset; at 0 evolution has switched dishabituation off."""
        keep = 1.0 - self.lifelike_params["dishab_relief"]
        for n in list(self._adapt):
            a = self._adapt[n] * keep
            if a > lifelike.PRUNE_EPS:
                self._adapt[n] = a
            else:
                del self._adapt[n]

    def _plasticity_snapshot(self) -> dict:
        """Lifelike snapshot keys for the learned layer. One stats pass
        covers all three fields (l1 == delta_norm)."""
        st = self.brain.plasticity_stats()
        return {"plasticity_delta": st["l1"],
                "plasticity_capped": st["capped"],
                "plasticity_edges": st["edges"]}

    def _habituation_snapshot(self) -> dict:
        """L1 of the adapted baselines — how nose-blind the worm currently
        is. Stuck at 0 across a lineage means adapt_rate evolved to 0 (the
        NES switched habituation off); ever-high means the worm never
        escapes its own patch. Keys are already sorted (see _chemo_pulse),
        so the sum is order-stable."""
        return {"habituation": round(sum(self._adapt.values()), 3)}

    def motion_stats(self) -> dict:
        """Spin diagnostics for /api/worms: revolutions and net drift over
        the last sim-minute. Measured baseline (seed 3, stock): a normal
        undulating worm shows turns_min ~14 (heading wobbles with the body
        wave) and drift_min ~400. The spinner signature is DRIFT, not turns:
        drift_min < ~50 world units while turns_min stays >= baseline =
        rotating in place (dish is 1600x1000)."""
        turns = sum(self._turn_window) / (2 * math.pi)
        if len(self._pos_window) >= 2:
            (x0, y0), (x1, y1) = self._pos_window[0], self._pos_window[-1]
            drift = math.hypot(x1 - x0, y1 - y0)
        else:
            drift = 0.0
        return {"turns_min": round(turns, 2), "drift_min": round(drift, 1)}

    def lifelike_payload(self) -> dict:
        """Every lifelike key a public payload carries, gated per feature —
        the ONE owner of which keys exist when. snapshot(), the focus
        broadcast, and /api/worms all spread this verbatim; a feature wired
        at only some sites would make the viewer and the REST API silently
        disagree (habituation touched three call sites before this existed).
        Empty dict with everything off, so default-off payloads are
        unchanged byte for byte. (The overview tray deliberately carries
        satiety/dead only — it doesn't use this.)"""
        out: dict = {}
        if self._hunger_on:
            out["satiety"] = round(self.satiety, 3)
            out["dead"] = self.dead
        if self._plasticity_on:
            out.update(self._plasticity_snapshot())
        if self._habituation_on:
            out.update(self._habituation_snapshot())
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
            # Lifelike keys appear only when the features are on; the
            # per-feature gating has one owner (lifelike_payload).
            **self.lifelike_payload(),
        }
