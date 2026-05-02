"""Simulation state: positions, velocities, bonds, parameters."""
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
    k_bend: float = 8.0#4.0

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
    bp_break_length: float = 1.8#2.4  # bonds break if stretched beyond this
    bp_form_prob: float = 0.6     # per-attempt formation probability
    bp_break_rate: float = 0.002  # per-attempt baseline break probability
    bp_min_separation: int = 4    # minimum |i-j| along chain to be base-pair eligible
    bonding_interval: int = 10    # steps between bond-update passes


@dataclass
class SimState:
    bases: np.ndarray              # (N,) int8
    pos: np.ndarray                # (N, 2) float64
    vel: np.ndarray                # (N, 2) float64
    force: np.ndarray              # (N, 2) float64
    backbone: np.ndarray           # (N-1, 2) int   pairs (i, i+1)
    base_pairs: np.ndarray         # (M, 2) int    dynamic; M grows/shrinks
    bp_partner: np.ndarray         # (N,) int      partner index or -1
    params: SimParams = field(default_factory=SimParams)
    step_count: int = 0

    @classmethod
    def from_sequence(cls, seq: str, params: SimParams | None = None) -> "SimState":
        params = params or SimParams()
        bases = encode(seq)
        n = len(bases)

        # Stack beads just left of the portal, ready to extrude.
        # Index 0 is the leading bead (closest to portal mouth, emerges first).
        x0 = params.portal_x - 0.5
        spacing = params.r_back
        pos = np.zeros((n, 2), dtype=np.float64)
        pos[:, 0] = x0 - np.arange(n) * spacing
        # Tiny y-jitter so backbone doesn't sit exactly collinear (helps WCA kick in).
        rng = np.random.default_rng(42)
        pos[:, 1] = rng.normal(0.0, 0.02, size=n)

        vel = np.zeros((n, 2), dtype=np.float64)
        force = np.zeros((n, 2), dtype=np.float64)

        backbone = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int32)
        base_pairs = np.zeros((0, 2), dtype=np.int32)
        bp_partner = -np.ones(n, dtype=np.int32)

        return cls(
            bases=bases,
            pos=pos,
            vel=vel,
            force=force,
            backbone=backbone,
            base_pairs=base_pairs,
            bp_partner=bp_partner,
            params=params,
        )

    @property
    def n(self) -> int:
        return len(self.bases)
