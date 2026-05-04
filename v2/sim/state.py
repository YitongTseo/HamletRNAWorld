"""Simulation state: positions, velocities, bonds, parameters.

Supports multiple polymer chains in a single SimState. All chains share the
same particle arrays (concatenated); a `chain_id` mask records which bead
belongs to which chain so backbone bonds and bending triplets can be built
per-chain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sequence import encode


@dataclass
class SimParams:
    # Particle / interaction scales (LJ-style reduced units).
    sigma: float = 1.0          # WCA / bead size
    epsilon: float = 1.0        # WCA strength
    mass: float = 1.0

    # Backbone harmonic spring (bond length).
    k_back: float = 80.0
    r_back: float = 1.0         # equilibrium backbone bond length

    # Backbone bending stiffness (Kratky-Porod / WLC: V = k_bend * (1 + cos θ),
    # minimized when consecutive backbone bonds are collinear, θ = π).
    # Persistence length ≈ k_bend / kT in units of the bond length.
    k_bend: float = 8.0

    # Base-pair harmonic spring.
    k_bp: float = 40.0
    r_bp: float = 1.05          # slightly longer than backbone — looks nice in viz

    # Thermostat.
    kT: float = 0.35
    gamma: float = 1.5          # Langevin friction
    dt: float = 0.005

    # Extrusion zone.
    portal_x: float = -10.0     # beads start packed left of this and emerge right
    extrusion_force: float = 2.0
    extrusion_zone: float = 2.0  # thickness of region in which drift force applies

    # Soft confining walls. Right wall + y walls only — the "tube" extends freely
    # off-screen to the left, so we don't constrain -x.
    box_x_max: float = 16.0
    box_y_half: float = 9.0
    k_wall: float = 5.0

    # Dynamic bonding.
    bp_form_cutoff: float = 1.4   # candidate pair must be within this distance
    bp_break_length: float = 1.8  # bonds break if stretched beyond this
    bp_form_prob: float = 0.6     # per-attempt formation probability
    bp_break_rate: float = 0.002  # per-attempt baseline break probability
    bp_min_separation: int = 4    # minimum |i-j| along chain to be base-pair eligible
    bonding_interval: int = 10    # steps between bond-update passes


@dataclass
class SimState:
    bases: np.ndarray              # (N,) int8
    chain_id: np.ndarray           # (N,) int — which chain each bead belongs to
    pos: np.ndarray                # (N, 2) float64
    vel: np.ndarray                # (N, 2) float64
    force: np.ndarray              # (N, 2) float64
    strain: np.ndarray             # (N,) float64 — per-bead bending strain (1+cosθ)
    backbone: np.ndarray           # (M, 2) int — within-chain adjacent pairs
    bend_triplets: np.ndarray      # (T, 3) int — within-chain (i-1, i, i+1) triplets
    base_pairs: np.ndarray         # (P, 2) int — dynamic; grows/shrinks
    bp_partner: np.ndarray         # (N,) int — partner index or -1
    params: SimParams = field(default_factory=SimParams)
    step_count: int = 0

    # ------------------------------------------------------------------

    @classmethod
    def from_sequence(cls, seq: str, params: SimParams | None = None) -> "SimState":
        return cls.from_sequences([seq], params)

    @classmethod
    def from_sequences(
        cls, sequences: list[str], params: SimParams | None = None
    ) -> "SimState":
        params = params or SimParams()
        if not sequences:
            raise ValueError("need at least one sequence")

        encoded = [encode(s) for s in sequences]
        chain_lens = np.array([len(e) for e in encoded], dtype=np.int32)
        chain_starts = np.concatenate([[0], np.cumsum(chain_lens[:-1])]).astype(np.int32)
        n = int(chain_lens.sum())

        bases = np.concatenate(encoded).astype(np.int8)
        chain_id = np.repeat(np.arange(len(sequences), dtype=np.int32), chain_lens)

        # Within-chain backbone bonds and bending triplets.
        backbone = []
        triplets = []
        for c, (start, length) in enumerate(zip(chain_starts, chain_lens)):
            for i in range(length - 1):
                backbone.append((start + i, start + i + 1))
            for i in range(length - 2):
                triplets.append((start + i, start + i + 1, start + i + 2))
        backbone_arr = np.array(backbone or [(0, 0)], dtype=np.int32)
        if not backbone:
            backbone_arr = np.zeros((0, 2), dtype=np.int32)
        triplets_arr = np.array(triplets, dtype=np.int32) if triplets else np.zeros((0, 3), dtype=np.int32)

        # All beads stack in a single queue at the portal mouth, in the order:
        # chain 0 (closest to portal, emerges first) then chain 1, etc.
        x0 = params.portal_x - 0.5
        spacing = params.r_back
        pos = np.zeros((n, 2), dtype=np.float64)
        pos[:, 0] = x0 - np.arange(n) * spacing
        rng = np.random.default_rng(42)
        pos[:, 1] = rng.normal(0.0, 0.02, size=n)

        vel = np.zeros((n, 2), dtype=np.float64)
        force = np.zeros((n, 2), dtype=np.float64)
        strain = np.zeros(n, dtype=np.float64)
        base_pairs = np.zeros((0, 2), dtype=np.int32)
        bp_partner = -np.ones(n, dtype=np.int32)

        return cls(
            bases=bases,
            chain_id=chain_id,
            pos=pos,
            vel=vel,
            force=force,
            strain=strain,
            backbone=backbone_arr,
            bend_triplets=triplets_arr,
            base_pairs=base_pairs,
            bp_partner=bp_partner,
            params=params,
        )

    @property
    def n(self) -> int:
        return len(self.bases)

    @property
    def n_chains(self) -> int:
        return int(self.chain_id.max()) + 1 if len(self.chain_id) else 0
