"""Glyph atlas: rasterize a TTF/OTF font into a single luminance texture.

For each character in `chars`, we draw the glyph into one cell of a grid
texture and remember its UV box. The atlas is built once at startup and
uploaded as a single 8-bit grayscale GL texture; the renderer samples each
letter bead's quad from its glyph cell.

Conventions
- The image is flipped vertically before being handed back so callers can
  upload it directly with OpenGL UV conventions (v=0 at bottom).
- UV boxes are stored as (uv_min, uv_max) where uv_min is bottom-left and
  uv_max is top-right of the cell, matching `mix(uv_min, uv_max, t)` where
  t = corner + 0.5 in the vertex shader.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_FONT_CANDIDATES: list[str] = [
    # Menlo first — tighter, more typewriter-y feel than Courier New.
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/Library/Fonts/Courier New.ttf",
]


def _find_font(candidates: list[str]) -> str:
    for p in candidates:
        if Path(p).exists():
            return p
    raise FileNotFoundError(
        "No suitable monospace font found. Pass --font /path/to/font.ttf"
    )


@dataclass
class FontAtlas:
    image: Image.Image          # PIL 'L' mode image, already vertically flipped
    width: int
    height: int
    advance_world: float        # one-character pitch in world units
    cell_pixel_h: int
    char_uv: dict               # str -> ((u_min, v_min), (u_max, v_max))

    def uv_for(self, char: str) -> tuple[tuple[float, float], tuple[float, float]]:
        if char in self.char_uv:
            return self.char_uv[char]
        # Unknown / missing: return a zero-area box so the quad samples blank.
        return ((0.0, 0.0), (0.0, 0.0))

    def texture_bytes(self) -> bytes:
        return self.image.tobytes()


def build_font_atlas(
    chars: str,
    font_path: str | None = None,
    font_size: int = 96,
    padding: int = 4,
    advance_world: float = 1.0,
) -> FontAtlas:
    """Build a luminance atlas of `chars` rendered with the given font."""
    font_path = font_path or _find_font(DEFAULT_FONT_CANDIDATES)
    font = ImageFont.truetype(font_path, font_size)

    chars = "".join(sorted(set(chars)))
    if not chars:
        raise ValueError("chars must be non-empty")

    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    max_adv = max(font.getlength(c) for c in chars)
    cell_w = int(max_adv) + 2 * padding
    cell_h = int(line_h) + 2 * padding

    n = len(chars)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    atlas_w = cell_w * cols
    atlas_h = cell_h * rows

    img = Image.new("L", (atlas_w, atlas_h), 0)
    draw = ImageDraw.Draw(img)

    char_uv: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    for i, c in enumerate(chars):
        col = i % cols
        row = i // cols
        cell_x = col * cell_w
        cell_y = row * cell_h
        # Center the glyph horizontally inside its cell.
        adv = font.getlength(c)
        x = cell_x + padding + (max_adv - adv) / 2
        y = cell_y + padding
        draw.text((x, y), c, fill=255, font=font)

        # UV in OpenGL convention (v=0 at bottom). After the vertical flip
        # below, the cell's top in PIL becomes its top in UV space too.
        uv_min = (cell_x / atlas_w, (atlas_h - cell_y - cell_h) / atlas_h)
        uv_max = ((cell_x + cell_w) / atlas_w, (atlas_h - cell_y) / atlas_h)
        char_uv[c] = (uv_min, uv_max)

    img = ImageOps.flip(img)
    return FontAtlas(
        image=img,
        width=atlas_w,
        height=atlas_h,
        advance_world=advance_world,
        cell_pixel_h=cell_h,
        char_uv=char_uv,
    )
