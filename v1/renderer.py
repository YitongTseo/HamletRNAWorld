"""
Pygame renderer for HamletRNAWorld.

Visual design:
  - Words are rendered as rotated text ON each bond segment (words = edges).
  - Invisible pivot joints connect consecutive words.
  - Each semantic word gets an "electron orbital" halo centred on its bond midpoint.
  - Dashed blue lines show intra-strand semantic attraction pairs.
  - Dotted amber lines show intra-strand repulsion pairs (subtle).
  - Small pivot dots mark the fold points (can be set to radius 0 to hide).
"""

import math
import pygame
import numpy as np
from typing import Dict, Optional, Tuple, List

from config import (
    WINDOW_PX, WINDOW_PY,
    BG_COLOR,
    HALO_RINGS, HALO_ALPHA_BASE, HALO_SCALE,
    FONT_PREFERENCE, FONT_SIZE_SEMANTIC, FONT_SIZE_FILLER,
    ATTRACT_LINE_THRESHOLD, ATTRACT_LINE_MAX_DIST,
    PIVOT_DOT_RADIUS, PIVOT_DOT_COLOR,
    MAX_EMIT_STRANDS,
)


# ── Utilities ──────────────────────────────────────────────────────────────

def _lerp(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _find_font(size: int) -> pygame.font.Font:
    for name in FONT_PREFERENCE:
        path = pygame.font.match_font(name)
        if path:
            return pygame.font.Font(path, size)
    return pygame.font.SysFont("serif", size)


def _draw_dashed_line(surf, color, p1, p2, dash=7, gap=5, width=1):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    pos, drawing = 0.0, True
    while pos < length:
        end = min(pos + (dash if drawing else gap), length)
        if drawing:
            s = (int(p1[0] + ux * pos),  int(p1[1] + uy * pos))
            e = (int(p1[0] + ux * end),  int(p1[1] + uy * end))
            pygame.draw.line(surf, color, s, e, width)
        pos, drawing = end, not drawing


# ── Main renderer ──────────────────────────────────────────────────────────

class Renderer:

    # Per-strand colour palette (RGB)
    STRAND_PALETTE = [
        (242, 115,  51),
        ( 51, 199, 235),
        (225,  56, 140),
        ( 89, 230, 102),
        (235, 217,  51),
        (153,  77, 242),
        (242, 166,  51),
        ( 64, 133, 242),
        (230, 140, 191),
        (102, 230, 191),
    ]

    def __init__(self):
        pygame.init()
        self.W, self.H = WINDOW_PX, WINDOW_PY
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("HamletRNAWorld")

        self.font_sem  = _find_font(FONT_SIZE_SEMANTIC)
        self.font_fill = _find_font(FONT_SIZE_FILLER)
        self.font_hud  = _find_font(12)

        # Alpha surface for halos and dashed lines
        self.alpha_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        self.clock = pygame.time.Clock()

        # Cache: (word, color_tuple) → unrotated pygame.Surface
        self._text_cache: Dict[str, pygame.Surface] = {}

    # ── Helpers ────────────────────────────────────────────────────────────

    def w2s(self, wx, wy) -> Tuple[int, int]:
        """World [0,1]² → screen pixels (flip Y: world y=0 is bottom)."""
        return int(wx * self.W), int((1.0 - wy) * self.H)

    def strand_color(self, sid: int) -> tuple:
        return self.STRAND_PALETTE[sid % len(self.STRAND_PALETTE)]

    def _word_surf(self, word: str, color: tuple, is_filler: bool) -> pygame.Surface:
        key = f"{word}|{'F' if is_filler else 'S'}|{color}"
        if key not in self._text_cache:
            font = self.font_fill if is_filler else self.font_sem
            self._text_cache[key] = font.render(word, True, color)
        return self._text_cache[key]

    def word_bbox(self, word: str, is_filler: bool = False) -> Tuple[int, int]:
        font = self.font_fill if is_filler else self.font_sem
        return font.size(word)

    # ── Bond midpoints ────────────────────────────────────────────────────
    # Each bond carries a word.  We build a list of
    #   (mx, my, angle_deg, word, is_filler, sem_idx, strand_id)
    # once per frame, then all draw passes use it.

    def _build_bond_words(self, n_bonds, bond_i_np, bond_j_np,
                          bond_strand_np, pos_np, meta, sem_idx_np):
        bw = []
        for k in range(n_bonds):
            ii, jj = int(bond_i_np[k]), int(bond_j_np[k])
            if ii >= len(pos_np) or jj >= len(pos_np):
                continue
            pi, pj = pos_np[ii], pos_np[jj]

            # Skip bonds crossing the torus boundary (would look wrong)
            if abs(pi[0] - pj[0]) > 0.5 or abs(pi[1] - pj[1]) > 0.5:
                continue

            sx1, sy1 = self.w2s(pi[0], pi[1])
            sx2, sy2 = self.w2s(pj[0], pj[1])
            mx, my   = (sx1 + sx2) // 2, (sy1 + sy2) // 2

            dx, dy = sx2 - sx1, sy2 - sy1
            # Angle for pygame (CCW positive; screen Y is already flipped via w2s)
            angle = math.degrees(math.atan2(-dy, dx))

            bm   = meta.get(jj, {})
            word = bm.get("word", "")
            if not word or bm.get("is_cap", False):
                continue
            is_filler = bm.get("is_filler", True)
            sidx      = int(sem_idx_np[jj]) if jj < len(sem_idx_np) else -1
            sid       = int(bond_strand_np[k])

            bw.append((mx, my, angle, word, is_filler, sidx, sid,
                       sx1, sy1, sx2, sy2))   # endpoints for pivot dots
        return bw

    # ── Draw: strand row separators (subtle guide lines) ──────────────────

    def draw_row_guides(self, n_active: int):
        """Faint horizontal rules between strand rows — only before strands fill in."""
        if n_active > 0:
            return
        from config import SPAWN_Y_POSITIONS
        for wy in SPAWN_Y_POSITIONS:
            sy = int((1.0 - wy) * self.H)
            pygame.draw.line(self.alpha_surf, (40, 40, 60, 60),
                             (20, sy), (self.W - 20, sy), 1)

    # ── Draw: electron orbital halos ──────────────────────────────────────

    def draw_halos(self, bond_words, pca_colors):
        for mx, my, angle, word, is_filler, sidx, sid, *_ in bond_words:
            if is_filler:
                continue
            base = pca_colors.get(word, (160, 160, 220))
            bw, bh = self.word_bbox(word, is_filler=False)
            half_w = bw // 2

            for ring in range(HALO_RINGS, 0, -1):
                frac  = ring / HALO_RINGS
                r     = int(half_w * HALO_SCALE * frac + bh * 0.5)
                alpha = int(HALO_ALPHA_BASE * (1.0 - frac * 0.55))
                col   = _lerp(base, (255, 255, 255), 0.2 * (1 - frac))
                pygame.draw.circle(self.alpha_surf, (*col, alpha), (mx, my), r)

            # Crisp inner ring outline
            pygame.draw.circle(self.alpha_surf,
                                (*base, 70), (mx, my),
                                int(half_w * 0.85 + bh * 0.3), 1)

    # ── Draw: semantic attraction/repulsion dashed lines ──────────────────

    def draw_semantic_lines(self, bond_words, sim_table):
        """Connect word midpoints within the same strand by dashed lines."""
        n = len(bond_words)
        for i in range(n):
            mx1, my1, _, w1, f1, si, sid1, *_ = bond_words[i]
            if si < 0 or f1:
                continue
            for j in range(i + 1, n):
                mx2, my2, _, w2, f2, sj, sid2, *_ = bond_words[j]
                if sj < 0 or f2:
                    continue
                if sid1 != sid2:
                    continue   # intra-strand only

                dx, dy = mx1 - mx2, my1 - my2
                dist_px = math.hypot(dx, dy)
                dist_w  = dist_px / self.W   # approx world distance

                if dist_w > ATTRACT_LINE_MAX_DIST:
                    continue

                cos_sim = float(sim_table[si, sj]) if (
                    si < sim_table.shape[0] and sj < sim_table.shape[1]
                ) else 0.0
                if abs(cos_sim) < ATTRACT_LINE_THRESHOLD:
                    continue

                prox     = max(0.0, 1.0 - dist_w / ATTRACT_LINE_MAX_DIST)
                strength = abs(cos_sim) * prox

                if cos_sim < 0:
                    # Attraction — blue-violet dashed
                    alpha = int(min(230, 50 + strength * 280))
                    col   = (int(80 + strength * 50),
                             int(110 + strength * 30),
                             int(230 + strength * 25),
                             alpha)
                    lw = max(1, int(strength * 3.5))
                    _draw_dashed_line(self.alpha_surf, col,
                                      (mx1, my1), (mx2, my2),
                                      dash=7, gap=4, width=lw)
                else:
                    # Repulsion — amber dotted (very subtle)
                    alpha = int(min(90, 12 + strength * 90))
                    col   = (220, 150, 40, alpha)
                    _draw_dashed_line(self.alpha_surf, col,
                                      (mx1, my1), (mx2, my2),
                                      dash=2, gap=7, width=1)

    # ── Draw: word text (rotated, centred on bond midpoint) ───────────────

    def draw_words(self, bond_words, pca_colors, pos_np, n_active):
        """Draw each word as rotated text at its bond midpoint.
           Also draw tiny pivot dots at joint positions."""

        # First pass: pivot dots
        if PIVOT_DOT_RADIUS > 0 and n_active > 0:
            for i in range(n_active):
                sx, sy = self.w2s(float(pos_np[i][0]), float(pos_np[i][1]))
                pygame.draw.circle(self.screen, PIVOT_DOT_COLOR, (sx, sy),
                                   PIVOT_DOT_RADIUS)

        # Second pass: word text — skip fillers entirely (invisible)
        for mx, my, angle, word, is_filler, sidx, sid, *_ in bond_words:
            if is_filler:
                continue   # filler joints are truly invisible
            else:
                pca = pca_colors.get(word, (180, 180, 220))
                sc  = self.strand_color(sid)
                color = _lerp(pca, sc, 0.30)
                color = tuple(min(255, int(c * 1.25 + 30)) for c in color)
                surf = self._word_surf(word, color, is_filler=False)

            rotated = pygame.transform.rotate(surf, angle)
            rect    = rotated.get_rect(center=(mx, my))
            self.screen.blit(rotated, rect)

    # ── Draw: HUD ─────────────────────────────────────────────────────────

    def draw_hud(self, stats, strand_preview, paused, speed):
        lines = [
            f"Words:   {stats['n']}",
            f"Strands: {stats['ns']}",
            f"Step:    {stats.get('step', 0)}",
            f"Speed: {speed}×" + ("  [PAUSED]" if paused else ""),
        ]
        x, y = 12, 12
        for line in lines:
            s = self.font_hud.render(line, True, (140, 140, 165))
            self.screen.blit(s, (x, y))
            y += 17

        if strand_preview:
            s = self.font_hud.render(f"→ {strand_preview}", True, (165, 150, 100))
            self.screen.blit(s, (12, self.H - 22))

        hint = "Space=pause  +/-=speed  R=reset  Q=quit"
        s = self.font_hud.render(hint, True, (55, 55, 75))
        self.screen.blit(s, (self.W - s.get_width() - 10, self.H - 20))

    # ── Full frame ────────────────────────────────────────────────────────

    def render_frame(self, n, n_bonds, pos_np, meta,
                     bond_i_np, bond_j_np, bond_strand_np,
                     sim_table, sem_idx_np, strand_id_np, chain_idx_np,
                     pca_colors, stats, strand_preview, paused, speed):

        self.screen.fill(BG_COLOR)
        self.alpha_surf.fill((0, 0, 0, 0))

        # Build per-bond word data (midpoints, angles, labels)
        bond_words = []
        if n > 0 and n_bonds > 0:
            bond_words = self._build_bond_words(
                n_bonds, bond_i_np, bond_j_np,
                bond_strand_np, pos_np, meta, sem_idx_np,
            )

        # ── Alpha-surface passes ───────────────────────────────────────────
        self.draw_row_guides(n)

        if bond_words:
            self.draw_semantic_lines(bond_words, sim_table)
            self.draw_halos(bond_words, pca_colors)

        # ── Blit alpha surface ─────────────────────────────────────────────
        self.screen.blit(self.alpha_surf, (0, 0))

        # ── Screen-space passes ────────────────────────────────────────────
        if bond_words:
            self.draw_words(bond_words, pca_colors, pos_np, n)

        self.draw_hud(stats, strand_preview, paused, speed)

        pygame.display.flip()
        self.clock.tick(60)
