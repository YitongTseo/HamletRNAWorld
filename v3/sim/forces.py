"""Force computation: WCA repulsion, harmonic bonds, extrusion drift, soft walls."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .state import SimState

# WCA cutoff = 2^(1/6) sigma (where attractive part of LJ would begin).
WCA_CUTOFF_FACTOR = 2.0 ** (1.0 / 6.0)


def compute_forces(s: SimState) -> None:
    """Recompute forces in place on s.force. Also refreshes s.strain."""
    p = s.params
    s.force.fill(0.0)
    s.strain.fill(0.0)

    _wca(s, p)
    _harmonic_pairs(s.pos, s.backbone, p.k_back, p.r_back, s.force)
    if len(s.base_pairs):
        _harmonic_pairs(s.pos, s.base_pairs, p.k_bp, p.r_bp, s.force)
    if len(s.bend_triplets) > 0:
        _backbone_bending(s, p)
    _extrusion_drift(s, p)
    _soft_walls(s, p)
    # Drag is now applied kinematically inside World.step (pos+vel snapped),
    # so no per-step drag force is needed. Removing it eliminates the
    # spring-oscillation instability that made dragged words jet away.


def _wca(s: SimState, p) -> None:
    """Repulsive WCA on near-neighbour pairs only — O(N) for our chain geometry.

    With ~1000 letter-beads queued in a long line off-screen, an O(N²)
    pairwise distance matrix is the bottleneck. cKDTree.query_pairs returns
    only those pairs within `r_cut` of each other; backbone-bonded pairs
    are then masked out (those interactions are handled by the harmonic
    spring instead).
    """
    pos = s.pos
    n = pos.shape[0]
    if n < 2:
        return
    sigma = p.sigma
    r_cut = WCA_CUTOFF_FACTOR * sigma
    eps = p.epsilon

    tree = cKDTree(pos)
    pair_arr = tree.query_pairs(r=r_cut, output_type="ndarray")
    if pair_arr.size == 0:
        return

    # Exclude backbone-bonded pairs (the harmonic spring handles those).
    if len(s.backbone):
        bb = s.backbone
        back_keys = bb[:, 0].astype(np.int64) * n + bb[:, 1].astype(np.int64)
        pair_keys = pair_arr[:, 0].astype(np.int64) * n + pair_arr[:, 1].astype(np.int64)
        keep = ~np.isin(pair_keys, back_keys)
        pair_arr = pair_arr[keep]
        if pair_arr.size == 0:
            return

    i_idx = pair_arr[:, 0]
    j_idx = pair_arr[:, 1]
    dr = pos[j_idx] - pos[i_idx]              # (M, 2)
    r2 = (dr * dr).sum(axis=1)
    safe = r2 > 1e-12
    inv_r2 = np.where(safe, 1.0 / np.where(safe, r2, 1.0), 0.0)
    sr2 = sigma * sigma * inv_r2
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    coeff = 24.0 * eps * (2.0 * sr12 - sr6) * inv_r2
    coeff = coeff * safe.astype(np.float64)

    # F_i = -dV/dr_i = coeff * (pos[i] - pos[j]) = -coeff * dr.
    f = -coeff[:, None] * dr
    np.add.at(s.force, i_idx, f)
    np.add.at(s.force, j_idx, -f)


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

    For each triplet (a, b, c) where b is the central (vertex) bead, θ is the
    angle at b. Minimum at θ = π (straight). Linear-in-cosine form is the
    worm-like-chain convention and is numerically stable everywhere (no
    1/sin θ singularity).

    Also writes per-bead bend strain (1 + cos θ ∈ [0, 2]) into s.strain[b].
    Endpoints of each chain stay at strain = 0 (no triplet defined there).
    """
    pos = s.pos
    triplets = s.bend_triplets
    k_per = s.bend_k                  # (T,) per-triplet stiffness
    a_idx = triplets[:, 0]
    b_idx = triplets[:, 1]
    c_idx = triplets[:, 2]

    e1 = pos[a_idx] - pos[b_idx]       # (T, 2)  bond from b -> a
    e2 = pos[c_idx] - pos[b_idx]       # (T, 2)  bond from b -> c
    n1 = np.linalg.norm(e1, axis=1)
    n2 = np.linalg.norm(e2, axis=1)
    safe = (n1 > 1e-9) & (n2 > 1e-9)
    n1_safe = np.where(safe, n1, 1.0)
    n2_safe = np.where(safe, n2, 1.0)
    e1h = e1 / n1_safe[:, None]
    e2h = e2 / n2_safe[:, None]
    cos_t = np.clip((e1h * e2h).sum(axis=1), -1.0, 1.0)

    # V = k*(1+cos θ)   =>  dV/d(cos θ) = k. k varies per triplet
    # (rigid inside a token, floppy at token boundaries).
    k_col = k_per[:, None]
    f_a = -k_col * (e2h - cos_t[:, None] * e1h) / n1_safe[:, None]
    f_c = -k_col * (e1h - cos_t[:, None] * e2h) / n2_safe[:, None]
    f_b = -(f_a + f_c)
    f_a[~safe] = 0.0
    f_b[~safe] = 0.0
    f_c[~safe] = 0.0

    np.add.at(s.force, a_idx, f_a)
    np.add.at(s.force, b_idx, f_b)
    np.add.at(s.force, c_idx, f_c)

    # Per-vertex strain (kept geometric — independent of stiffness — so the
    # strain visualization shows the actual bend angle).
    s.strain[b_idx] = 1.0 + cos_t


def _extrusion_drift(s: SimState, p) -> None:
    """Leftward push on beads inside the extrusion zone, with a per-letter
    angular offset so they emerge fanned out instead of straight.

    Each letter's `extrusion_angle` is a fixed ± jitter relative to a
    pure-left direction, set deterministically at SimState construction.
    A force of magnitude `extrusion_force` is applied at that angle for any
    bead with x > portal_x - extrusion_zone (i.e., still in or near the tube).
    """
    x = s.pos[:, 0]
    in_zone = x > (p.portal_x - p.extrusion_zone)
    if not in_zone.any():
        return
    angles = s.extrusion_angle[in_zone]
    f = p.extrusion_force
    s.force[in_zone, 0] -= f * np.cos(angles)
    s.force[in_zone, 1] -= f * np.sin(angles)


def _soft_walls(s: SimState, p) -> None:
    """Quadratic restoring force on the LEFT wall + both y walls.

    No right wall — beads queueing in "the tube" extend off-screen to +x freely.
    """
    k = p.k_wall
    under_x = s.pos[:, 0] - p.box_x_min
    s.force[:, 0] -= k * np.where(under_x < 0, under_x, 0.0)
    over_y_top = s.pos[:, 1] - p.box_y_half
    over_y_bot = s.pos[:, 1] + p.box_y_half
    s.force[:, 1] -= k * np.where(over_y_top > 0, over_y_top, 0.0)
    s.force[:, 1] -= k * np.where(over_y_bot < 0, over_y_bot, 0.0)
