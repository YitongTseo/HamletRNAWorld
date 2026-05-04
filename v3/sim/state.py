"""Simulation state for letter-bead, token-grouped polymer chains.

Beads are individual *characters*. Tokens (words + punctuation) are groups
of consecutive letter beads with a shared embedding and reactivity flag.

Physics is at the LETTER level (positions, velocities, backbone bonds,
pair bonds). Affinity gating for pair-bond formation looks up the token of
each letter and uses TOKEN-level cosine-distance affinity. This way:

- Long words physically take more chain length than short words.
- Words bend rigidly internally; major bends happen at token boundaries.
- A word with 5 letters can pick up to 5 pair partners, possibly with
  different other words → multi-partner bonding emerges naturally.
- Stop words and punctuation (token.reactive == False) are inert.
- Whitespace beads (token_id == -1) are inert and rendered as invisible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .strand import Strand, compute_token_affinity, token_embeddings_to_colors


@dataclass
class SimParams:
    sigma: float = 1.0
    epsilon: float = 1.0
    mass: float = 1.0

    # Backbone harmonic spring (bond length is uniform — one character pitch).
    k_back: float = 80.0
    r_back: float = 1.0

    # Bending stiffness. Two values: high inside a single token, low at
    # boundaries (where the chain wants to fold).
    k_bend_word: float = 40.0
    k_bend_inter: float = 1.5

    # Pair-bond ("hydrogen-style") spring.
    k_bp: float = 40.0
    r_bp: float = 1.05

    # Cooler + heavier-damped so the per-letter jitter stays calm. The system
    # still explores conformations (kT > 0) but with less high-frequency
    # twitchiness that makes type hard to read.
    kT: float = 0.18
    gamma: float = 3.0
    dt: float = 0.005

    # Portal sits on the right of the visible area; beads queue further right
    # (off-screen) and drift LEFTWARD as they emerge — that way the first letter
    # of each word is the leftmost on screen, so the strand reads naturally
    # left-to-right as it grows.
    portal_x: float = 10.0
    extrusion_force: float = 4.0
    extrusion_zone: float = 2.0
    # Random ejection angle (degrees, ±) added to the leftward base direction.
    # Per-letter, deterministic per bead so motion stays coherent over time.
    extrusion_angle_jitter_deg: float = 30.0
    # Empty space (world units) between consecutive sentences in the queue.
    # The drift's terminal velocity v_t = extrusion_force / gamma; a gap of
    # `g` translates to `g / v_t` sim-time units of wait between the last
    # letter of sentence k and the first letter of sentence k+1 emerging.
    inter_chain_gap: float = 8.0

    # Left wall + y walls. The right side is the "tube" and extends freely.
    box_x_min: float = -16.0
    box_y_half: float = 9.0
    k_wall: float = 5.0

    # Dynamic bonding.
    bp_form_cutoff: float = 1.4
    bp_break_length: float = 1.8
    bp_form_prob: float = 0.6
    bp_break_rate: float = 0.002
    # Sequence-separation gate is now in *letters*; with monospace, this is
    # roughly "letters must be at least this many slots apart along the chain".
    bp_min_separation: int = 6
    bonding_interval: int = 10

    # Affinity gate. Pairs below this threshold never bond. With a quadratic
    # exponent (default 2.0), the formation probability is
    #     p = bp_form_prob * affinity^bp_form_exponent
    # so weak-affinity pairs become MUCH rarer than they were under the
    # previous linear scaling.
    affinity_threshold: float = 0.7
    bp_form_exponent: float = 2.0


@dataclass
class SimState:
    # ----- letter-level (length N) -----
    chars: list[str]
    chain_id: np.ndarray              # (N,) int — which strand each letter belongs to
    token_id: np.ndarray              # (N,) int — global token index, or -1 for spaces
    colors: np.ndarray                # (N, 3) f32 — per-letter render color (=parent token color)

    pos: np.ndarray                   # (N, 2) f64
    vel: np.ndarray                   # (N, 2) f64
    force: np.ndarray                 # (N, 2) f64
    strain: np.ndarray                # (N,) f64
    extrusion_angle: np.ndarray       # (N,) f64 — per-letter ejection angle (rad)

    backbone: np.ndarray              # (M, 2) int — within-strand consecutive letters
    bend_triplets: np.ndarray         # (T, 3) int
    bend_k: np.ndarray                # (T,) f32 — per-triplet bending stiffness
    base_pairs: np.ndarray            # (P, 2) int
    bp_partner: np.ndarray            # (N,) int — letter partner or -1

    # ----- token-level (length T_total across all strands) -----
    tokens: list[str]
    token_embeddings: np.ndarray      # (T_total, D) f32
    token_affinity: np.ndarray        # (T_total, T_total) f32 in [0, 1]
    token_reactive: np.ndarray        # (T_total,) bool
    token_color: np.ndarray           # (T_total, 3) f32

    # External drag set by the UI. When drag_indices is non-empty, forces.py
    # applies a strong spring from each indexed letter toward drag_targets[i].
    drag_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    drag_targets: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float64))
    drag_k: float = 250.0

    params: SimParams = field(default_factory=SimParams)
    step_count: int = 0

    # ------------------------------------------------------------------

    @classmethod
    def from_strands(
        cls, strands: list[Strand], params: SimParams | None = None
    ) -> "SimState":
        params = params or SimParams()
        if not strands:
            raise ValueError("need at least one strand")

        # Concatenate token-level data. We re-index per-strand token_ids into
        # a single global token table.
        tokens: list[str] = []
        embeddings: list[np.ndarray] = []
        reactive_flags: list[np.ndarray] = []
        token_offsets = []
        for s in strands:
            token_offsets.append(len(tokens))
            tokens.extend(s.tokens)
            embeddings.append(s.embeddings)
            reactive_flags.append(s.reactive)
        token_embeddings = np.concatenate(embeddings, axis=0).astype(np.float32) if embeddings else np.zeros((0, 1), dtype=np.float32)
        token_reactive = np.concatenate(reactive_flags) if reactive_flags else np.zeros(0, dtype=bool)
        token_affinity = compute_token_affinity(token_embeddings)
        token_color = token_embeddings_to_colors(token_embeddings)

        # Concatenate letter-level data.
        chars: list[str] = []
        chain_id_parts: list[np.ndarray] = []
        token_id_parts: list[np.ndarray] = []
        letter_color_parts: list[np.ndarray] = []
        chain_lens: list[int] = []
        for c, s in enumerate(strands):
            chars.extend(s.chars)
            chain_id_parts.append(np.full(s.n_letters, c, dtype=np.int32))
            # Re-index this strand's local token_ids into the global table.
            local_ids = s.token_ids
            global_ids = np.where(local_ids >= 0, local_ids + token_offsets[c], -1).astype(np.int32)
            token_id_parts.append(global_ids)
            # Per-letter color = parent token's color, or a neutral gray for spaces.
            colors = np.tile(np.array([0.45, 0.45, 0.55], dtype=np.float32), (s.n_letters, 1))
            for li, gid in enumerate(global_ids):
                if gid >= 0:
                    colors[li] = token_color[gid]
            letter_color_parts.append(colors)
            chain_lens.append(s.n_letters)

        chain_id = np.concatenate(chain_id_parts)
        token_id = np.concatenate(token_id_parts)
        letter_colors = np.concatenate(letter_color_parts, axis=0).astype(np.float32)
        n = sum(chain_lens)

        # Backbone bonds — within each chain only. Bending triplets — within
        # each chain only. Per-triplet k_bend depends on whether the triplet
        # is entirely inside one token (high k) or crosses a boundary (low k).
        backbone: list[tuple[int, int]] = []
        triplets: list[tuple[int, int, int]] = []
        bend_ks: list[float] = []
        chain_starts = np.concatenate([[0], np.cumsum(np.array(chain_lens[:-1], dtype=np.int32))]).astype(np.int32)
        for c, length in enumerate(chain_lens):
            start = int(chain_starts[c])
            for i in range(length - 1):
                backbone.append((start + i, start + i + 1))
            for i in range(length - 2):
                a = start + i
                b = start + i + 1
                cc = start + i + 2
                triplets.append((a, b, cc))
                # Stiff if all three letters share the SAME non-spacer token.
                ta = int(token_id[a])
                tb = int(token_id[b])
                tc = int(token_id[cc])
                same_token = (ta >= 0) and (ta == tb) and (tb == tc)
                bend_ks.append(params.k_bend_word if same_token else params.k_bend_inter)

        backbone_arr = np.array(backbone, dtype=np.int32) if backbone else np.zeros((0, 2), dtype=np.int32)
        triplets_arr = np.array(triplets, dtype=np.int32) if triplets else np.zeros((0, 3), dtype=np.int32)
        bend_k_arr = np.array(bend_ks, dtype=np.float32) if bend_ks else np.zeros((0,), dtype=np.float32)

        # Initial positions: queue at the portal mouth, with a configurable gap
        # between consecutive chains. The drift's terminal velocity is uniform
        # across the chain, so a spatial gap between strands translates to a
        # time-wait between them emerging at the portal mouth.
        x0 = params.portal_x + 0.5
        spacing = params.r_back
        gap = params.inter_chain_gap
        pos = np.zeros((n, 2), dtype=np.float64)
        cursor = 0.0
        for c, length in enumerate(chain_lens):
            start = int(chain_starts[c])
            for i in range(length):
                pos[start + i, 0] = x0 + cursor + i * spacing
            cursor += length * spacing
            if c < len(chain_lens) - 1:
                cursor += gap
        rng = np.random.default_rng(42)
        pos[:, 1] = rng.normal(0.0, 0.02, size=n)

        # Per-letter ejection angle, deterministic. Uniform in [-jitter, +jitter].
        jitter_rad = float(np.deg2rad(params.extrusion_angle_jitter_deg))
        extrusion_angle = rng.uniform(-jitter_rad, jitter_rad, size=n)

        return cls(
            chars=chars,
            chain_id=chain_id,
            token_id=token_id,
            colors=letter_colors,
            pos=pos,
            vel=np.zeros((n, 2), dtype=np.float64),
            force=np.zeros((n, 2), dtype=np.float64),
            strain=np.zeros(n, dtype=np.float64),
            extrusion_angle=extrusion_angle,
            backbone=backbone_arr,
            bend_triplets=triplets_arr,
            bend_k=bend_k_arr,
            base_pairs=np.zeros((0, 2), dtype=np.int32),
            bp_partner=-np.ones(n, dtype=np.int32),
            tokens=tokens,
            token_embeddings=token_embeddings,
            token_affinity=token_affinity,
            token_reactive=token_reactive,
            token_color=token_color,
            params=params,
        )

    @property
    def n(self) -> int:
        return len(self.chars)

    @property
    def n_chains(self) -> int:
        return int(self.chain_id.max()) + 1 if len(self.chain_id) else 0

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)
