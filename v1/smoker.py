"""
Emission logic — places each RNA strand as a straight horizontal line.

All joints are materialised at once (no bead-by-bead streaming).
Each strand is centred at x=0.5 at its designated y-row.

Strand layout for ["sleep","dream","death","rub"]:
  cap ── sleep ── dream ── death ── rub
   0       1        2        3       4    (chain_idx)

Bond 0→1 labelled "sleep", bond 1→2 "dream", etc.
The cap joint is invisible and carries no word.
"""

import numpy as np
from typing import List, Optional

from config import (
    SPAWN_Y_POSITIONS, WORLD_W,
    BOND_REST_BASE, BOND_REST_CHAR, BOND_REST,
    MAX_EMIT_STRANDS,
)
from corpus import Token


class BlackSmoker:
    def __init__(
        self,
        strands: List[List[Token]],
        embed_store,
        rng: Optional[np.random.Generator] = None,
    ):
        self.strands = strands
        self.store   = embed_store
        self.rng     = rng or np.random.default_rng(42)

        self._emit_order = list(range(min(MAX_EMIT_STRANDS, len(strands))))
        self.strands_emitted = 0
        self.done = False

        # preview text for HUD
        self._last_preview: Optional[List[Token]] = None

    @property
    def active_strand_preview(self) -> Optional[List[Token]]:
        return self._last_preview

    def step(self, sim_step: int) -> bool:
        """
        Emit all remaining strands immediately on the first few calls.
        Returns True once all strands have been placed.
        """
        if self.done:
            return True

        if self.strands_emitted >= len(self._emit_order):
            self.done = True
            self._last_preview = None
            return True

        idx = self._emit_order[self.strands_emitted]
        strand = self.strands[idx]
        y_row  = SPAWN_Y_POSITIONS[self.strands_emitted % len(SPAWN_Y_POSITIONS)]

        self._place_strand_straight(strand, y_row)
        self._last_preview = strand
        self.strands_emitted += 1

        if self.strands_emitted >= len(self._emit_order):
            self.done = True
            self._last_preview = None

        return self.done

    # ── Internal ────────────────────────────────────────────────────────────

    def _place_strand_straight(self, strand: List[Token], cy: float):
        """
        Materialise a strand as a perfectly straight horizontal line
        centred at (0.5, cy).  All joints start with zero velocity.
        """
        import simulation as sim

        nb = int(sim.n_active[None])
        ns = int(sim.n_strands[None])

        if nb + len(strand) + 1 > sim.MAX_BEADS or ns >= sim.MAX_STRANDS:
            print(f"  [smoker] capacity exceeded — skipping strand")
            return

        # ── Build bead list: [cap, word_0, word_1, ..., word_N-1] ──────────
        beads = []

        # Cap (silent start joint)
        beads.append({
            "word": "", "raw": "", "is_filler": True,
            "sem_idx": -1, "is_cap": True,
        })

        for tok in strand:
            sidx = -1 if tok.is_filler else self.store.word_to_idx.get(tok.word, -1)
            beads.append({
                "word":      tok.word,
                "raw":       tok.raw,
                "is_filler": tok.is_filler,
                "sem_idx":   sidx,
                "is_cap":    False,
            })

        # ── Compute bond rest lengths and total chain length ────────────────
        rest_lens = []   # one per bond (beads[k-1] → beads[k])
        for k in range(1, len(beads)):
            w = beads[k]["word"]
            rest = BOND_REST_BASE + BOND_REST_CHAR * len(w)
            rest_lens.append(rest)

        total_len = sum(rest_lens)

        # ── Positions along a horizontal line centred at x=0.5 ─────────────
        x_start = 0.5 - total_len / 2.0
        # Clamp so we don't fall off-screen (world is toroidal but we want visibility)
        x_start = max(0.04, min(0.96 - total_len, x_start))

        xs = [x_start]
        for r in rest_lens:
            xs.append(xs[-1] + r)

        # ── Reserve strand slot ─────────────────────────────────────────────
        sim.strand_start[ns]      = nb
        sim.strand_len[ns]        = len(beads)
        sim.n_strands[None]       = ns + 1

        # ── Write bead data ─────────────────────────────────────────────────
        for k, bead in enumerate(beads):
            bi = nb + k
            x  = xs[k]
            # Tiny random perturbation so physics doesn't start in degenerate state
            x += self.rng.normal(0, 0.0005)
            y  = cy + self.rng.normal(0, 0.0005)

            sim.pos[bi]        = [x, y]
            sim.vel[bi]        = [0.0, 0.0]
            sim.force[bi]      = [0.0, 0.0]
            sim.strand_id[bi]  = ns
            sim.chain_idx[bi]  = k
            sim.sem_idx[bi]    = bead["sem_idx"]
            sim._bead_meta[bi] = bead

            # Register bond  (prev joint → this joint)
            if k > 0:
                bnd = int(sim.n_bonds[None])
                if bnd < sim.MAX_BONDS:
                    sim.bond_i[bnd]        = bi - 1
                    sim.bond_j[bnd]        = bi
                    sim.bond_strand[bnd]   = ns
                    sim.bond_rest_len[bnd] = rest_lens[k - 1]
                    sim.n_bonds[None]      = bnd + 1

        sim.n_active[None] = nb + len(beads)
        print(f"  Placed strand {ns}: "
              f"{' '.join(b['word'] for b in beads if b['word'])} "
              f"  len={total_len:.3f}  y={cy:.2f}")
