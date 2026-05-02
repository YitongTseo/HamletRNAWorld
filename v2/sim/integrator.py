"""Velocity-Verlet integrator with Langevin (Ornstein-Uhlenbeck) thermostat."""
from __future__ import annotations

import numpy as np

from .forces import compute_forces
from .state import SimState


class Integrator:
    def __init__(self, state: SimState, seed: int = 0):
        self.s = state
        self._rng = np.random.default_rng(seed)
        compute_forces(self.s)

    def step(self) -> None:
        s = self.s
        p = s.params
        m = p.mass
        dt = p.dt

        # Velocity Verlet: half-kick, drift, recompute forces, half-kick.
        s.vel += 0.5 * dt * s.force / m
        s.pos += dt * s.vel
        compute_forces(s)
        s.vel += 0.5 * dt * s.force / m

        # Ornstein-Uhlenbeck thermostat step (applied after VV, "OBABO"-style tail).
        c1 = np.exp(-p.gamma * dt)
        c2 = np.sqrt((1.0 - c1 * c1) * p.kT / m)
        s.vel = c1 * s.vel + c2 * self._rng.standard_normal(s.vel.shape)

        s.step_count += 1
