"""
HamletRNAWorld — main entry point.

Controls:
    Space   — pause / unpause
    +/-     — speed up / slow down (steps per frame)
    R       — reset
    Q / Esc — quit
"""

import numpy as np
import taichi as ti

# ── Taichi init (before importing simulation) ──────────────────────────────
try:
    ti.init(arch=ti.gpu, default_fp=ti.f32)
    print("Taichi: GPU (Metal)")
except Exception:
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    print("Taichi: CPU fallback")

import pygame
import simulation as sim

from config import (
    WINDOW_PX, WINDOW_PY, SMOKER_X, SMOKER_Y, MAX_BEADS, TEST_MODE,
)
from corpus import load_strands, load_test_strands, unique_semantic_words
from embeddings import EmbeddingStore
from smoker import BlackSmoker
from renderer import Renderer


def _build_smoker_and_place(strands, store, rng):
    """Create a smoker and emit all strands immediately."""
    smoker = BlackSmoker(strands, store, rng)
    step = 0
    # Emit all strands (each call to step places one strand)
    while not smoker.done:
        smoker.step(step)
        step += 1
    return smoker


def run():
    # ── Corpus + embeddings ───────────────────────────────────────────────
    if TEST_MODE:
        strands = load_test_strands()
    else:
        strands = load_strands()

    words = unique_semantic_words(strands)
    print(f"Building embedding store for {len(words)} semantic words…")
    store = EmbeddingStore(words)

    print("Computing PCA colours…")
    pca_colors = store.compute_pca_colors()

    # ── Simulation ────────────────────────────────────────────────────────
    sim.attach_sim_table(store.sim_table)
    sim.init()
    sim.init_smoker_glow(SMOKER_X, SMOKER_Y)

    rng = np.random.default_rng(0)
    smoker = _build_smoker_and_place(strands, store, rng)
    sim.refresh_render_colors()

    # ── Renderer ──────────────────────────────────────────────────────────
    renderer = Renderer()

    paused    = False
    step_mult = 1     # start slow — user can speed up with +
    sim_step  = 0

    print("\n=== Running — Space=pause  +/-=speed  R=reset  Q=quit ===\n")

    running = True
    while running:
        # ── Events ─────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif k in (pygame.K_EQUALS, pygame.K_PLUS):
                    step_mult = min(step_mult + 1, 30)
                    print(f"Speed: {step_mult}×")
                elif k == pygame.K_MINUS:
                    step_mult = max(step_mult - 1, 1)
                    print(f"Speed: {step_mult}×")
                elif k == pygame.K_r:
                    sim.init()
                    sim.init_smoker_glow(SMOKER_X, SMOKER_Y)
                    smoker = _build_smoker_and_place(strands, store, rng)
                    sim.refresh_render_colors()
                    sim_step  = 0
                    step_mult = 1
                    print("Reset.")

        # ── Physics ────────────────────────────────────────────────────────
        if not paused:
            for _ in range(step_mult):
                sim.step()
                sim_step += 1

        # ── Collect render data ────────────────────────────────────────────
        n  = int(sim.n_active[None])
        nb = int(sim.n_bonds[None])
        ns = int(sim.n_strands[None])
        meta = sim._bead_meta

        if n > 0:
            pos_np       = sim.pos.to_numpy()[:n]
            sem_idx_np   = sim.sem_idx.to_numpy()[:n]
            strand_id_np = sim.strand_id.to_numpy()[:n]
            chain_idx_np = sim.chain_idx.to_numpy()[:n]
        else:
            pos_np = sem_idx_np = strand_id_np = chain_idx_np = np.zeros((0,))

        if nb > 0:
            bond_i_np      = sim.bond_i.to_numpy()[:nb]
            bond_j_np      = sim.bond_j.to_numpy()[:nb]
            bond_strand_np = sim.bond_strand.to_numpy()[:nb]
        else:
            bond_i_np = bond_j_np = bond_strand_np = np.zeros((0,), dtype=np.int32)

        stats = {
            "n": n, "ns": ns, "nb": nb,
            "emitted": smoker.strands_emitted,
            "step": sim_step,
        }

        # ── Render ─────────────────────────────────────────────────────────
        renderer.render_frame(
            n=n, n_bonds=nb,
            pos_np=pos_np, meta=meta,
            bond_i_np=bond_i_np,
            bond_j_np=bond_j_np,
            bond_strand_np=bond_strand_np,
            sim_table=store.sim_table,
            sem_idx_np=sem_idx_np,
            strand_id_np=strand_id_np,
            chain_idx_np=chain_idx_np,
            pca_colors=pca_colors,
            stats=stats,
            strand_preview=None,
            paused=paused,
            speed=step_mult,
        )

    pygame.quit()
    print("Done.")


if __name__ == "__main__":
    run()
