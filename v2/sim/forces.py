"""Force computation: WCA repulsion, harmonic bonds, extrusion drift, soft walls."""
from __future__ import annotations

import numpy as np

from .state import SimState

# WCA cutoff = 2^(1/6) sigma (where attractive part of LJ would begin).
WCA_CUTOFF_FACTOR = 2.0 ** (1.0 / 6.0)


def compute_forces(s: SimState) -> None:
    """Recompute forces in place on s.force."""
    p = s.params
    s.force.fill(0.0)

    _wca(s, p)
    _harmonic_pairs(s.pos, s.backbone, p.k_back, p.r_back, s.force)
    if len(s.base_pairs):
        _harmonic_pairs(s.pos, s.base_pairs, p.k_bp, p.r_bp, s.force)
    if p.k_bend > 0.0 and s.n >= 3:
        _backbone_bending(s, p)
    _extrusion_drift(s, p)
    _soft_walls(s, p)


def _wca(s: SimState, p) -> None:
    pos = s.pos
    n = pos.shape[0]
    sigma = p.sigma
    r_cut = WCA_CUTOFF_FACTOR * sigma
    eps = p.epsilon

    # Pairwise displacements: dr[i,j] = pos[j] - pos[i]
    dr = pos[None, :, :] - pos[:, None, :]                  # (N, N, 2)
    r2 = np.einsum("ijk,ijk->ij", dr, dr)                   # (N, N)
    np.fill_diagonal(r2, np.inf)                            # ignore self-pairs
    # Exclude adjacent backbone neighbors — the backbone harmonic spring already
    # determines their interaction; WCA on top would over-constrain the spacing.
    idx = np.arange(n - 1)
    r2[idx, idx + 1] = np.inf
    r2[idx + 1, idx] = np.inf
    mask = r2 < r_cut * r_cut
    if not mask.any():
        return

    inv_r2 = np.zeros_like(r2)
    inv_r2[mask] = 1.0 / r2[mask]
    sr2 = (sigma * sigma) * inv_r2                          # (sigma/r)^2
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    # Force magnitude / r:  24 eps (2 sr12 - sr6) / r^2
    coeff = np.zeros_like(r2)
    coeff[mask] = 24.0 * eps * (2.0 * sr12[mask] - sr6[mask]) * inv_r2[mask]
    # Force on i from j is -dV/dr_i  =  coeff * (pos[i] - pos[j])  =  -coeff * dr[i,j]
    f_pairs = -coeff[:, :, None] * dr                       # (N, N, 2)
    s.force += f_pairs.sum(axis=1)


def _harmonic_pairs(pos, pairs, k, r0, force_out) -> None:
    """Apply harmonic spring V = 0.5 k (r-r0)^2 over an (M,2) list of index pairs."""
    if len(pairs) == 0:
        return
    i = pairs[:, 0]
    j = pairs[:, 1]
    dr = pos[j] - pos[i]                                    # (M, 2), points i -> j
    r = np.sqrt((dr * dr).sum(axis=1))                      # (M,)
    safe = r > 1e-9
    r_safe = np.where(safe, r, 1.0)
    # F_i = -dV/dr_i * grad_i(r) = -k*(r-r0) * (-(j-i)/r) = +k*(r-r0)/r * (j-i)
    # So when r>r0 (stretched), F_i points along +dr (toward j) — attractive. Correct.
    f_mag = k * (r - r0)
    f = (f_mag / r_safe)[:, None] * dr                      # force on i
    f[~safe] = 0.0
    np.add.at(force_out, i, f)
    np.add.at(force_out, j, -f)


def _backbone_bending(s: SimState, p) -> None:
    """3-body bending potential V = k_bend * (1 + cos θ) over each backbone triplet.

    For each triplet (i-1, i, i+1) along the chain, θ is the angle at bead i.
    Minimum at θ = π (straight). Linear-in-cosine form is the worm-like-chain
    convention and is numerically stable everywhere (no 1/sin θ singularity).
    """
    pos = s.pos
    k = p.k_bend
    # Triplets along backbone: a = i-1, b = i (vertex), c = i+1.
    a = pos[:-2]
    b = pos[1:-1]
    c = pos[2:]
    e1 = a - b                         # (M, 2)  bond from b -> a
    e2 = c - b                         # (M, 2)  bond from b -> c
    n1 = np.linalg.norm(e1, axis=1)
    n2 = np.linalg.norm(e2, axis=1)
    safe = (n1 > 1e-9) & (n2 > 1e-9)
    n1_safe = np.where(safe, n1, 1.0)
    n2_safe = np.where(safe, n2, 1.0)
    e1h = e1 / n1_safe[:, None]
    e2h = e2 / n2_safe[:, None]
    cos_t = np.clip((e1h * e2h).sum(axis=1), -1.0, 1.0)

    # V = k*(1+cos θ)   =>  dV/d(cos θ) = k
    # F_a = -k * (e2h - cos_t * e1h) / |e1|
    # F_c = -k * (e1h - cos_t * e2h) / |e2|
    # F_b = -(F_a + F_c)
    f_a = -k * (e2h - cos_t[:, None] * e1h) / n1_safe[:, None]
    f_c = -k * (e1h - cos_t[:, None] * e2h) / n2_safe[:, None]
    f_b = -(f_a + f_c)
    f_a[~safe] = 0.0
    f_b[~safe] = 0.0
    f_c[~safe] = 0.0

    n = s.n
    np.add.at(s.force, np.arange(0, n - 2), f_a)
    np.add.at(s.force, np.arange(1, n - 1), f_b)
    np.add.at(s.force, np.arange(2, n),     f_c)


def _extrusion_drift(s: SimState, p) -> None:
    """Constant +x push on any bead inside the extrusion zone near the portal mouth."""
    x = s.pos[:, 0]
    in_zone = x < (p.portal_x + p.extrusion_zone)
    s.force[in_zone, 0] += p.extrusion_force


def _soft_walls(s: SimState, p) -> None:
    """Quadratic restoring force on the right wall and both y walls.

    No left wall — beads queueing in "the tube" extend off-screen to -x freely.
    """
    k = p.k_wall
    over_x = s.pos[:, 0] - p.box_x_max
    s.force[:, 0] -= k * np.where(over_x > 0, over_x, 0.0)
    over_y_top = s.pos[:, 1] - p.box_y_half
    over_y_bot = s.pos[:, 1] + p.box_y_half
    s.force[:, 1] -= k * np.where(over_y_top > 0, over_y_top, 0.0)
    s.force[:, 1] -= k * np.where(over_y_bot < 0, over_y_bot, 0.0)
