"""
Taichi-based 2D physics engine for the RNA-world simulation.

Each bead represents one word in a Hamlet sentence.
  - Bonded interactions: harmonic spring + angle bending stiffness
  - Non-bonded interactions: semantically-modulated Lennard-Jones
      cos_sim > 0  (similar words)  → repulsion dominates
      cos_sim < 0  (dissimilar)     → attraction well → secondary structure
  - Brownian motion (random thermal kicks each step)
  - Toroidal boundary conditions
  - Filler words: excluded from non-bonded interactions (neutral beads)

Rendering note
--------------
Taichi 1.7.x canvas.circles() / canvas.lines() require *Taichi Vector.field*
arguments, not raw numpy arrays.  All render buffers are therefore allocated
as Taichi fields here; inactive slots are placed off-screen at (-2, -2) so a
single full-buffer draw call safely skips them.
"""

import numpy as np
import taichi as ti
from config import (
    MAX_BEADS, MAX_STRANDS, MAX_STRAND_LEN,
    WORLD_W, WORLD_H,
    BEAD_MASS, BEAD_RADIUS,
    BOND_K, BOND_REST, BOND_DAMP, BEND_K,
    LJ_EPSILON, LJ_SIGMA, LJ_CUTOFF,
    KT, DT, DAMPING,
    BOND_REST_BASE, BOND_REST_CHAR,
)

# ── Physics fields ──────────────────────────────────────────────────────────
pos      = ti.Vector.field(2, dtype=ti.f32, shape=MAX_BEADS)
vel      = ti.Vector.field(2, dtype=ti.f32, shape=MAX_BEADS)
force    = ti.Vector.field(2, dtype=ti.f32, shape=MAX_BEADS)

strand_id  = ti.field(dtype=ti.i32, shape=MAX_BEADS)   # -1 = inactive
chain_idx  = ti.field(dtype=ti.i32, shape=MAX_BEADS)   # position within strand
sem_idx    = ti.field(dtype=ti.i32, shape=MAX_BEADS)   # -1 for filler/inactive

n_active   = ti.field(dtype=ti.i32, shape=())

strand_start = ti.field(dtype=ti.i32, shape=MAX_STRANDS)
strand_len   = ti.field(dtype=ti.i32, shape=MAX_STRANDS)
n_strands    = ti.field(dtype=ti.i32, shape=())

# Similarity table — placeholder replaced by attach_sim_table()
sim_table  = ti.field(dtype=ti.f32, shape=(1, 1))

# ── Bond topology (populated by smoker as beads are added) ──────────────────
# Each bond is an (i, j) pair of bead indices, consecutive in chain.
MAX_BONDS    = MAX_BEADS
bond_i       = ti.field(dtype=ti.i32, shape=MAX_BONDS)
bond_j       = ti.field(dtype=ti.i32, shape=MAX_BONDS)
bond_strand  = ti.field(dtype=ti.i32, shape=MAX_BONDS)
# Per-bond rest length (set from word character counts in smoker)
bond_rest_len = ti.field(dtype=ti.f32, shape=MAX_BONDS)
n_bonds      = ti.field(dtype=ti.i32, shape=())

# ── Render fields (Taichi Vector.field — required by canvas API) ────────────
# Beads: inactive placed at (-2,-2) = off-screen
render_color = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BEADS)

# Bonds: each bond k → two vertices at [2k] and [2k+1]
# Inactive bond verts stay at (-2,-2)
bond_render_verts  = ti.Vector.field(2, dtype=ti.f32, shape=MAX_BONDS * 2)
bond_render_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_BONDS * 2)

# Smoker glow — N_GLOW separate single-element fields so each ring can have
# a distinct radius in a canvas.circles() call.
N_GLOW = 5
# Lists of single-element Taichi fields, one per ring
smoker_glow_pos    = [ti.Vector.field(2, dtype=ti.f32, shape=1) for _ in range(N_GLOW)]
smoker_glow_color  = [ti.Vector.field(3, dtype=ti.f32, shape=1) for _ in range(N_GLOW)]
smoker_glow_radii  = []  # plain Python list of float

# Python-side bead metadata (word, is_filler per bead index)
_bead_meta: dict = {}


# ── Sim table attachment ────────────────────────────────────────────────────

def attach_sim_table(np_table: np.ndarray):
    global sim_table
    n = np_table.shape[0]
    sim_table = ti.field(dtype=ti.f32, shape=(n, n))
    sim_table.from_numpy(np_table.astype(np.float32))


# ── Torus helpers ───────────────────────────────────────────────────────────

@ti.func
def torus_delta(a: ti.template(), b: ti.template()) -> ti.template():
    d = b - a
    if d[0] >  WORLD_W * 0.5: d[0] -= WORLD_W
    if d[0] < -WORLD_W * 0.5: d[0] += WORLD_W
    if d[1] >  WORLD_H * 0.5: d[1] -= WORLD_H
    if d[1] < -WORLD_H * 0.5: d[1] += WORLD_H
    return d


@ti.func
def wrap(p: ti.template()) -> ti.template():
    p[0] = p[0] % WORLD_W
    p[1] = p[1] % WORLD_H
    return p


# ── Force kernels ───────────────────────────────────────────────────────────

@ti.kernel
def clear_forces():
    for i in range(n_active[None]):
        force[i] = ti.Vector([0.0, 0.0])


@ti.kernel
def apply_bond_forces():
    nb = n_bonds[None]
    for k in range(nb):
        i    = bond_i[k]
        j    = bond_j[k]
        rest = bond_rest_len[k]
        delta   = torus_delta(pos[i], pos[j])
        dist    = delta.norm() + 1e-9
        unit    = delta / dist
        stretch = dist - rest
        f = BOND_K * stretch * unit
        rel_v = vel[j] - vel[i]
        f += BOND_DAMP * rel_v.dot(unit) * unit
        force[i] +=  f
        force[j] += -f


@ti.kernel
def apply_bending_forces():
    for s in range(n_strands[None]):
        start  = strand_start[s]
        length = strand_len[s]
        for k in range(1, length - 1):
            prev = start + k - 1
            curr = start + k
            nxt  = start + k + 1
            if (strand_id[prev] != s or strand_id[curr] != s
                    or strand_id[nxt] != s):
                continue
            d1   = torus_delta(pos[prev], pos[curr])
            d2   = torus_delta(pos[curr], pos[nxt])
            len1 = d1.norm() + 1e-9
            len2 = d2.norm() + 1e-9
            t1   = d1 / len1
            t2   = d2 / len2
            cross = t1[0] * t2[1] - t1[1] * t2[0]
            fb   = BEND_K * cross
            perp1 = ti.Vector([-t1[1],  t1[0]])
            perp2 = ti.Vector([-t2[1],  t2[0]])
            force[prev] -= fb / len1 * perp1
            force[nxt]  += fb / len2 * perp2
            force[curr] += fb * (perp1 / len1 - perp2 / len2)


@ti.kernel
def apply_nonbonded_forces():
    nb = n_active[None]
    for i in range(nb):
        si = sem_idx[i]
        if si < 0:
            continue
        for j in range(i + 1, nb):
            sj = sem_idx[j]
            if sj < 0:
                continue
            # Intra-strand only — skip inter-strand pairs entirely
            if strand_id[i] != strand_id[j]:
                continue
            # Skip adjacent bonded beads
            ki = chain_idx[i]
            kj = chain_idx[j]
            if ti.abs(ki - kj) <= 2:   # skip cap→word1 and immediate neighbors
                continue

            delta = torus_delta(pos[i], pos[j])
            dist  = delta.norm()
            if dist >= LJ_CUTOFF or dist < 1e-6:
                continue

            cos_sim = sim_table[si, sj]
            sr   = LJ_SIGMA / dist
            sr6  = sr * sr * sr * sr * sr * sr
            sr12 = sr6 * sr6

            f_mag = (4.0 * LJ_EPSILON / dist) * (
                -12.0 * sr12 - 6.0 * cos_sim * sr6
            )
            f_mag = ti.max(ti.min(f_mag, 5.0), -5.0)
            fvec  = f_mag * (delta / dist)
            force[i] +=  fvec
            force[j] -= fvec


@ti.kernel
def apply_brownian_and_integrate():
    nb = n_active[None]
    for i in range(nb):
        noise = ti.Vector([
            ti.randn(ti.f32),
            ti.randn(ti.f32),
        ]) * ti.sqrt(2.0 * KT / DT)
        acc    = (force[i] + noise) / BEAD_MASS
        vel[i] = vel[i] * DAMPING + acc * DT
        pos[i] = wrap(pos[i] + vel[i] * DT)


# ── Bond render update kernel ───────────────────────────────────────────────

@ti.kernel
def update_bond_render():
    """Copy current bead positions into the bond render vertex buffer."""
    nb = n_bonds[None]
    for k in range(nb):
        ii = bond_i[k]
        jj = bond_j[k]
        bond_render_verts[k * 2]     = pos[ii]
        bond_render_verts[k * 2 + 1] = pos[jj]


# ── Colour helpers (called from Python) ────────────────────────────────────

# Per-strand colour palette (must match main.py's _STRAND_PALETTE)
_PALETTE_NP = np.array([
    [0.95, 0.45, 0.20],
    [0.20, 0.78, 0.92],
    [0.88, 0.22, 0.55],
    [0.35, 0.90, 0.40],
    [0.92, 0.85, 0.20],
    [0.60, 0.30, 0.95],
    [0.95, 0.65, 0.20],
    [0.25, 0.52, 0.95],
    [0.90, 0.55, 0.75],
    [0.40, 0.90, 0.75],
], dtype=np.float32)

_FILLER_COLOR = np.array([0.38, 0.38, 0.43], dtype=np.float32)


def refresh_render_colors():
    """Rebuild render_color and bond colour arrays from current strand IDs."""
    n  = int(n_active[None])
    nb = int(n_bonds[None])
    if n == 0:
        return

    sid_np = strand_id.to_numpy()[:n]
    colors = np.zeros((MAX_BEADS, 3), dtype=np.float32)

    for i in range(n):
        sid = int(sid_np[i])
        if sid < 0:
            continue
        bm = _bead_meta.get(i, {})
        if bm.get("is_filler", True):
            colors[i] = _FILLER_COLOR
        else:
            colors[i] = _PALETTE_NP[sid % len(_PALETTE_NP)]

    render_color.from_numpy(colors)

    # Bond colours
    if nb > 0:
        bc = np.zeros((MAX_BONDS * 2, 3), dtype=np.float32)
        bsid = bond_strand.to_numpy()[:nb]
        for k in range(nb):
            c = _PALETTE_NP[int(bsid[k]) % len(_PALETTE_NP)] * 0.7
            bc[k * 2]     = c
            bc[k * 2 + 1] = c
        bond_render_colors.from_numpy(bc)


# ── Initialisation ──────────────────────────────────────────────────────────

def init():
    global _bead_meta
    _bead_meta = {}
    n_active[None]  = 0
    n_strands[None] = 0
    n_bonds[None]   = 0
    bond_rest_len.fill(BOND_REST)

    # Place ALL beads off-screen so a full-buffer draw skips them
    off = np.full((MAX_BEADS, 2), -2.0, dtype=np.float32)
    pos.from_numpy(off)

    # Place ALL bond verts off-screen
    off2 = np.full((MAX_BONDS * 2, 2), -2.0, dtype=np.float32)
    bond_render_verts.from_numpy(off2)

    # Zero colours
    render_color.from_numpy(np.zeros((MAX_BEADS, 3), dtype=np.float32))
    bond_render_colors.from_numpy(np.zeros((MAX_BONDS * 2, 3), dtype=np.float32))

    # Zero vel / force
    vel.fill(0.0)
    force.fill(0.0)

    sid_np  = np.full(MAX_BEADS, -1, dtype=np.int32)
    cidx_np = np.full(MAX_BEADS, -1, dtype=np.int32)
    sem_np  = np.full(MAX_BEADS, -1, dtype=np.int32)
    strand_id.from_numpy(sid_np)
    chain_idx.from_numpy(cidx_np)
    sem_idx.from_numpy(sem_np)


def init_smoker_glow(sx: float, sy: float):
    """Pre-build fixed smoker glow geometry into per-ring Taichi fields."""
    global smoker_glow_radii
    smoker_glow_radii = []
    for k in range(N_GLOW):
        t = (N_GLOW - k) / N_GLOW
        smoker_glow_pos[k].from_numpy(
            np.array([[sx, sy]], dtype=np.float32)
        )
        smoker_glow_color[k].from_numpy(
            np.array([[t, t * 0.25, t * 0.05]], dtype=np.float32)
        )
        smoker_glow_radii.append(0.010 * (k + 1) * 1.5)


# ── One simulation step ─────────────────────────────────────────────────────

def step():
    clear_forces()
    apply_bond_forces()
    apply_bending_forces()
    apply_nonbonded_forces()
    apply_brownian_and_integrate()
