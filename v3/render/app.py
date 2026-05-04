"""moderngl-window app: lays out letter-beads as glyphs over colored auras.

Visual stack (bottom → top):
  1. Aura — colored "electron cloud" point sprite at the bond-formation
     radius, only for reactive (content-word) letters.
  2. Bonds — backbone + dynamic pair lines.
  3. Letter glyph — sharp WHITE text rotated by its local backbone tangent.
  4. Pipe wall — opaque white rect with a black portal hole on the right;
     occludes letters still inside the tube.

UI:
  * Mouse wheel: zoom in/out (around screen center).
  * Right-click drag: pan camera.
  * Left-click on a letter: grab the whole word (token); the cursor pulls all
    its letters via a strong spring while the rest of the chain follows
    through the backbone.
  * Left-click on empty area: pan camera (so the user always has *some* way
    to move around regardless of pointer landing).
  * S / SPACE: toggle backbone strain coloring.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import moderngl
import moderngl_window as mglw
import numpy as np
from PIL import Image, ImageDraw

from sim.world import World
from .font_atlas import build_font_atlas

SHADER_DIR = Path(__file__).parent / "shaders"


def _ortho(left: float, right: float, bottom: float, top: float,
           near: float = -1.0, far: float = 1.0) -> np.ndarray:
    return np.array([
        2.0 / (right - left), 0.0, 0.0, 0.0,
        0.0, 2.0 / (top - bottom), 0.0, 0.0,
        0.0, 0.0, -2.0 / (far - near), 0.0,
        -(right + left) / (right - left),
        -(top + bottom) / (top - bottom),
        -(far + near) / (far - near),
        1.0,
    ], dtype="f4")


_QUAD_CORNERS = np.array([
    [-0.5, -0.5],
    [+0.5, -0.5],
    [-0.5, +0.5],
    [+0.5, +0.5],
], dtype=np.float32)


class HamletApp(mglw.WindowConfig):
    title = "Hamlet World v3 — letter beads"
    gl_version = (3, 3)
    window_size = (1600, 900)
    aspect_ratio = 16 / 9
    resizable = True
    samples = 4

    # Default world rectangle (the camera identity view).
    BASE_WORLD_X = (-16.0, 16.0)
    BASE_WORLD_Y = (-9.0, 9.0)
    GLYPH_WORLD_SIZE = 0.55
    AURA_WORLD_RADIUS = 5.0
    POSITION_SMOOTHING = 0.15
    SIM_STEPS_PER_FRAME = 8

    # Bond-label glyph size (small text drawn at each bond's midpoint).
    LABEL_GLYPH_SIZE = 0.30
    LABEL_OFFSET_FROM_BOND = 0.4
    MAX_LABEL_CHARS = 2000

    # PCA hover popup: small panel near cursor showing 2D PCA scatter of all
    # token embeddings, with the hovered word's dot lit up.
    POPUP_PIXEL_SIZE = 220
    POPUP_PIXEL_OFFSET = 18
    POPUP_DOT_RADIUS = 2          # grey background dots
    POPUP_HIGHLIGHT_RADIUS = 4    # bright hover dot

    # Pipe geometry: extends well to the right of the portal so it always
    # covers any zoomed-out view AND the long staggered queue of 50 sentences.
    PIPE_RIGHT_EXTENT = 4000.0
    PIPE_Y_HALF = 30.0
    PORTAL_RADIUS = 1.8

    # Drag pickup radius (world units).
    PICK_RADIUS = 0.7

    _WORLD: Optional[World] = None
    _FONT_PATH: Optional[str] = None

    # ------------------------------------------------------------------

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if HamletApp._WORLD is None:
            raise RuntimeError("call render.app.run(world)")
        self.world = HamletApp._WORLD
        s = self.world.state
        n = s.n
        self._show_strain = False

        # --- Camera state ---
        self._cam_x = 0.0
        self._cam_y = 0.0
        self._cam_zoom = 1.0
        self._proj_dirty = True

        # --- Drag state ---
        self._drag_letters = np.zeros(0, dtype=np.int32)
        self._drag_offsets = np.zeros((0, 2), dtype=np.float64)
        self._is_panning = False

        # --- Visible-letter mask (skips whitespace) ---
        is_ws = np.array([c.isspace() for c in s.chars], dtype=bool)
        self._visible_mask = (s.token_id >= 0) & ~is_ws
        self._visible_idx = np.where(self._visible_mask)[0].astype(np.int32)
        m = int(self._visible_idx.size)

        # --- Reactive-visible mask (for aura only) ---
        v_token = s.token_id[self._visible_idx]
        v_reactive = s.token_reactive[v_token]
        self._reactive_idx = self._visible_idx[v_reactive]
        m_reactive = int(self._reactive_idx.size)

        # --- Per-letter neighbor lookup for tangent ---
        prev_neighbor = np.full(m, -1, dtype=np.int32)
        next_neighbor = np.full(m, -1, dtype=np.int32)
        for k, i in enumerate(self._visible_idx):
            i = int(i)
            ci = int(s.chain_id[i])
            j = i - 1
            while j >= 0 and int(s.chain_id[j]) == ci:
                if not s.chars[j].isspace():
                    prev_neighbor[k] = j
                    break
                j -= 1
            j = i + 1
            while j < n and int(s.chain_id[j]) == ci:
                if not s.chars[j].isspace():
                    next_neighbor[k] = j
                    break
                j += 1
        self._prev_neighbor = prev_neighbor
        self._next_neighbor = next_neighbor

        # --- Font atlas ---
        # Include digits + '.' so we can also render bond-strength numbers.
        chars_needed = sorted(
            {s.chars[i] for i in self._visible_idx} | set("0123456789.")
        )
        self.atlas = build_font_atlas("".join(chars_needed),
                                      font_path=HamletApp._FONT_PATH)
        self.atlas_tex = self.ctx.texture(
            (self.atlas.width, self.atlas.height),
            components=1,
            data=self.atlas.texture_bytes(),
        )
        self.atlas_tex.repeat_x = False
        self.atlas_tex.repeat_y = False
        self.atlas_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # --- Shader programs ---
        self.letter_program = self.ctx.program(
            vertex_shader=(SHADER_DIR / "letter.vert").read_text(),
            fragment_shader=(SHADER_DIR / "letter.frag").read_text(),
        )
        self.bond_program = self.ctx.program(
            vertex_shader=(SHADER_DIR / "bond.vert").read_text(),
            fragment_shader=(SHADER_DIR / "bond.frag").read_text(),
        )
        self.aura_program = self.ctx.program(
            vertex_shader=(SHADER_DIR / "aura.vert").read_text(),
            fragment_shader=(SHADER_DIR / "aura.frag").read_text(),
        )
        self.pipe_program = self.ctx.program(
            vertex_shader=(SHADER_DIR / "pipe.vert").read_text(),
            fragment_shader=(SHADER_DIR / "pipe.frag").read_text(),
        )

        self.letter_program["u_glyph_size"].value = self.GLYPH_WORLD_SIZE
        self.bond_program["u_intensity"].value = 0.6
        self.bond_program["u_show_strain"].value = 0
        self._aura_pixel_size = 0.0

        # --- Smoothed render positions (EMA) ---
        self._smoothed_pos = s.pos.astype("f4").copy()

        # --- Letter glyph instance buffers ---
        uv_min = np.zeros((m, 2), dtype=np.float32)
        uv_max = np.zeros((m, 2), dtype=np.float32)
        for k, i in enumerate(self._visible_idx):
            uvmn, uvmx = self.atlas.uv_for(s.chars[int(i)])
            uv_min[k] = uvmn
            uv_max[k] = uvmx
        letter_colors = np.ones((m, 3), dtype=np.float32)  # white

        self.quad_corners_buf = self.ctx.buffer(_QUAD_CORNERS.tobytes())
        self.uv_min_buf = self.ctx.buffer(uv_min.tobytes())
        self.uv_max_buf = self.ctx.buffer(uv_max.tobytes())
        self.letter_color_buf = self.ctx.buffer(letter_colors.tobytes())
        self.letter_pos_buf = self.ctx.buffer(reserve=m * 2 * 4, dynamic=True)
        self.letter_tangent_buf = self.ctx.buffer(reserve=m * 2 * 4, dynamic=True)
        self.letter_vao = self.ctx.vertex_array(
            self.letter_program,
            [
                (self.quad_corners_buf, "2f", "in_corner"),
                (self.letter_pos_buf, "2f/i", "in_pos"),
                (self.letter_tangent_buf, "2f/i", "in_tangent"),
                (self.uv_min_buf, "2f/i", "in_uv_min"),
                (self.uv_max_buf, "2f/i", "in_uv_max"),
                (self.letter_color_buf, "3f/i", "in_color"),
            ],
        )

        # --- Bond-strength label glyph buffers (reuses letter_program) ---
        # Affinity numbers like "0.83" rendered above each pair-bond, oriented
        # along the bond direction. All buffers are dynamic per-frame.
        ml = self.MAX_LABEL_CHARS
        self.label_pos_buf = self.ctx.buffer(reserve=ml * 2 * 4, dynamic=True)
        self.label_tangent_buf = self.ctx.buffer(reserve=ml * 2 * 4, dynamic=True)
        self.label_uv_min_buf = self.ctx.buffer(reserve=ml * 2 * 4, dynamic=True)
        self.label_uv_max_buf = self.ctx.buffer(reserve=ml * 2 * 4, dynamic=True)
        self.label_color_buf = self.ctx.buffer(reserve=ml * 3 * 4, dynamic=True)
        self.label_vao = self.ctx.vertex_array(
            self.letter_program,
            [
                (self.quad_corners_buf, "2f", "in_corner"),
                (self.label_pos_buf, "2f/i", "in_pos"),
                (self.label_tangent_buf, "2f/i", "in_tangent"),
                (self.label_uv_min_buf, "2f/i", "in_uv_min"),
                (self.label_uv_max_buf, "2f/i", "in_uv_max"),
                (self.label_color_buf, "3f/i", "in_color"),
            ],
        )
        self._label_count = 0

        # --- Aura buffers ---
        self.aura_pos_buf = self.ctx.buffer(reserve=max(1, m_reactive) * 2 * 4, dynamic=True)
        if m_reactive > 0:
            self.aura_color_buf = self.ctx.buffer(s.colors[self._reactive_idx].astype("f4").tobytes())
        else:
            self.aura_color_buf = self.ctx.buffer(reserve=4)
        self.aura_vao = self.ctx.vertex_array(
            self.aura_program,
            [
                (self.aura_pos_buf, "2f", "in_pos"),
                (self.aura_color_buf, "3f", "in_color"),
            ],
        )
        self._n_reactive = m_reactive

        # --- Bond buffers ---
        max_segments = (n - 1) + (n // 2) + 4
        self.bond_pos_buf = self.ctx.buffer(reserve=max_segments * 2 * 2 * 4, dynamic=True)
        self.bond_kind_buf = self.ctx.buffer(reserve=max_segments * 2 * 4, dynamic=True)
        self.bond_strain_buf = self.ctx.buffer(reserve=max_segments * 2 * 4, dynamic=True)
        self.bond_vao = self.ctx.vertex_array(
            self.bond_program,
            [
                (self.bond_pos_buf, "2f", "in_pos"),
                (self.bond_kind_buf, "1i", "in_kind"),
                (self.bond_strain_buf, "1f", "in_strain"),
            ],
        )
        self._bond_vertex_count = 0
        self._n_visible = m

        # --- Pipe geometry buffer (static) ---
        portal_x = float(s.params.portal_x)
        pipe_x_min = portal_x
        pipe_x_max = portal_x + self.PIPE_RIGHT_EXTENT
        pipe_y_min = -self.PIPE_Y_HALF
        pipe_y_max = +self.PIPE_Y_HALF
        pipe_quad = np.array([
            [pipe_x_min, pipe_y_min],
            [pipe_x_max, pipe_y_min],
            [pipe_x_min, pipe_y_max],
            [pipe_x_max, pipe_y_max],
        ], dtype=np.float32)
        self.pipe_buf = self.ctx.buffer(pipe_quad.tobytes())
        self.pipe_vao = self.ctx.vertex_array(
            self.pipe_program, [(self.pipe_buf, "2f", "in_pos")]
        )
        self.pipe_program["u_portal"].value = (portal_x, 0.0)
        self.pipe_program["u_portal_r"].value = self.PORTAL_RADIUS
        self.pipe_program["u_pipe_color"].value = (0.93, 0.93, 0.96)
        self.pipe_program["u_pipe_x"].value = (pipe_x_min, pipe_x_max)
        self.pipe_program["u_pipe_y"].value = (pipe_y_min, pipe_y_max)

        # --- PCA hover popup (debugging panel) ---
        self._setup_pca_popup()

        # --- GL state ---
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        try:
            self.ctx.line_width = 2.0
        except Exception:
            pass

    # ------------------------------------------------------------------
    # PCA popup
    # ------------------------------------------------------------------

    def _setup_pca_popup(self) -> None:
        """Compute 2D PCA on token embeddings and pre-render the grey scatter
        background to a PIL image. The image is uploaded as a texture; on
        hover, we re-composite (background + bright dot) and write to the
        same texture without re-allocating."""
        s = self.world.state
        embeds = s.token_embeddings
        size = self.POPUP_PIXEL_SIZE

        if embeds.shape[0] == 0:
            self._popup_enabled = False
            return

        # 2D PCA via SVD (no extra deps).
        centered = embeds.astype(np.float32) - embeds.mean(axis=0, keepdims=True)
        if embeds.shape[0] >= 2:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            k = min(2, vt.shape[0])
            pca = centered @ vt[:k].T
            if k < 2:
                pca = np.concatenate([pca, np.zeros((pca.shape[0], 2 - k), dtype=np.float32)], axis=1)
        else:
            pca = np.zeros((embeds.shape[0], 2), dtype=np.float32)

        margin = 22
        inner = size - 2 * margin
        p_min = pca.min(axis=0)
        p_max = pca.max(axis=0)
        span = np.maximum(p_max - p_min, 1e-9)
        # Map PCA → pixel coords (PIL Y=0 at top).
        pca_px = np.zeros_like(pca)
        pca_px[:, 0] = margin + (pca[:, 0] - p_min[0]) / span[0] * inner
        # Flip Y so larger PC2 values are visually higher on screen.
        pca_px[:, 1] = margin + (1.0 - (pca[:, 1] - p_min[1]) / span[1]) * inner
        self._pca_coords_px = pca_px

        # Pre-render the static greyscale scatter once.
        self._popup_bg_image = self._render_popup_background()

        # Texture (RGBA, written each time hover changes).
        self._popup_tex = self.ctx.texture((size, size), components=4,
                                           data=self._popup_bg_image.tobytes())
        self._popup_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Popup shader + VAO. Position+UV interleaved; uploaded each frame
        # the popup is shown so it follows the cursor.
        self.popup_program = self.ctx.program(
            vertex_shader=(SHADER_DIR / "popup.vert").read_text(),
            fragment_shader=(SHADER_DIR / "popup.frag").read_text(),
        )
        # 4 vertices, each (pos2 + uv2) = 4 floats = 16 bytes.
        self.popup_quad_buf = self.ctx.buffer(reserve=4 * 4 * 4, dynamic=True)
        self.popup_vao = self.ctx.vertex_array(
            self.popup_program,
            [(self.popup_quad_buf, "2f 2f", "in_pos", "in_uv")],
        )

        self._hover_token = -1
        self._hover_token_cached = -2  # forces composite on first match
        self._popup_enabled = True

    def _render_popup_background(self) -> Image.Image:
        size = self.POPUP_PIXEL_SIZE
        img = Image.new("RGBA", (size, size), (16, 18, 30, 235))
        draw = ImageDraw.Draw(img)

        # Crosshairs through panel center (the "bullseye").
        cx, cy = size // 2, size // 2
        margin = 22
        cross = (90, 95, 130, 220)
        draw.line([(margin, cy), (size - margin, cy)], fill=cross, width=1)
        draw.line([(cx, margin), (cx, size - margin)], fill=cross, width=1)
        # Outer border
        draw.rectangle([(0, 0), (size - 1, size - 1)], outline=(110, 120, 160, 255))

        # Grey dots for every token.
        r = self.POPUP_DOT_RADIUS
        for x_px, y_px in self._pca_coords_px:
            draw.ellipse([(x_px - r, y_px - r), (x_px + r, y_px + r)],
                         fill=(150, 155, 185, 200))

        # Tiny label in the corner so the user knows what they're looking at.
        try:
            draw.text((6, 4), "PC0 / PC1", fill=(180, 185, 220, 230))
        except Exception:
            pass
        return img

    def _composite_popup(self, hover_token: int) -> None:
        img = self._popup_bg_image.copy()
        draw = ImageDraw.Draw(img)
        x_px, y_px = self._pca_coords_px[hover_token]
        # Glow rings → bright core for the highlighted dot.
        for r, alpha in [(11, 60), (8, 110), (5, 200)]:
            draw.ellipse([(x_px - r, y_px - r), (x_px + r, y_px + r)],
                         fill=(255, 230, 140, alpha))
        rr = self.POPUP_HIGHLIGHT_RADIUS
        draw.ellipse([(x_px - rr, y_px - rr), (x_px + rr, y_px + rr)],
                     fill=(255, 255, 255, 255))

        # Word label below the panel — a small tag identifying the hovered word.
        token_str = self.world.state.tokens[hover_token]
        try:
            draw.text((8, self.POPUP_PIXEL_SIZE - 16), token_str,
                      fill=(255, 255, 255, 240))
        except Exception:
            pass
        self._popup_tex.write(img.tobytes())

    def _render_popup(self) -> None:
        if not self._popup_enabled:
            return
        sx, sy = self._cursor_screen
        win_w, win_h = self.wnd.size
        size = self.POPUP_PIXEL_SIZE
        off = self.POPUP_PIXEL_OFFSET

        # Place panel top-left near cursor; flip if it would clip off-screen.
        x_left = sx + off
        y_top = sy + off
        if x_left + size > win_w:
            x_left = sx - size - off
        if y_top + size > win_h:
            y_top = sy - size - off
        x_left = max(0, min(win_w - size, x_left))
        y_top = max(0, min(win_h - size, y_top))
        x_right = x_left + size
        y_bot = y_top + size

        def to_ndc(px: float, py: float) -> tuple[float, float]:
            return (2.0 * px / win_w - 1.0, 1.0 - 2.0 * py / win_h)

        bl = to_ndc(x_left, y_bot)
        br = to_ndc(x_right, y_bot)
        tl = to_ndc(x_left, y_top)
        tr = to_ndc(x_right, y_top)
        # TRIANGLE_STRIP order: BL, BR, TL, TR. UV (0,1), (1,1), (0,0), (1,0).
        quad = np.array([
            [bl[0], bl[1], 0.0, 1.0],
            [br[0], br[1], 1.0, 1.0],
            [tl[0], tl[1], 0.0, 0.0],
            [tr[0], tr[1], 1.0, 0.0],
        ], dtype=np.float32)
        self.popup_quad_buf.write(quad.tobytes())

        self._popup_tex.use(location=0)
        self.popup_program["u_tex"].value = 0
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.popup_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE  # restore additive

    def _update_hover(self, sx: float, sy: float) -> None:
        self._cursor_screen = (sx, sy)
        wx, wy = self.screen_to_world(sx, sy)
        letter = self._pick_letter(wx, wy)
        if letter is not None and self.world.state.token_id[letter] >= 0:
            self._hover_token = int(self.world.state.token_id[letter])
        else:
            self._hover_token = -1

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _view_rect(self) -> tuple[float, float, float, float]:
        """Current camera view in world coords: (left, right, bottom, top)."""
        half_w = (self.BASE_WORLD_X[1] - self.BASE_WORLD_X[0]) / 2 / self._cam_zoom
        half_h = (self.BASE_WORLD_Y[1] - self.BASE_WORLD_Y[0]) / 2 / self._cam_zoom
        return (
            self._cam_x - half_w, self._cam_x + half_w,
            self._cam_y - half_h, self._cam_y + half_h,
        )

    def _update_projection(self) -> None:
        l, r, b, t = self._view_rect()
        proj = _ortho(l, r, b, t)
        for prog in (self.letter_program, self.bond_program, self.aura_program,
                     self.pipe_program):
            prog["u_proj"].write(proj.tobytes())

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        """Convert window pixel coordinates (Y from top) to world coordinates."""
        win_w, win_h = self.wnd.size
        l, r, b, t = self._view_rect()
        wx = l + (sx / win_w) * (r - l)
        # moderngl-window's mouse Y is from the top, so flip.
        wy = b + (1.0 - sy / win_h) * (t - b)
        return wx, wy

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def on_render(self, time: float, frame_time: float) -> None:
        self.world.step(self.SIM_STEPS_PER_FRAME)
        self._update_smoothed_positions()
        self._update_drag_targets()       # so the drag follows latest cursor
        self._upload_geometry()

        if self._proj_dirty:
            self._update_projection()
            self._proj_dirty = False

        # Aura sprite size scales with framebuffer + zoom.
        fb_w, _ = self.ctx.fbo.size
        l, r, _, _ = self._view_rect()
        new_aura_size = 2.0 * self.AURA_WORLD_RADIUS * (fb_w / (r - l))
        if new_aura_size != self._aura_pixel_size:
            self._aura_pixel_size = new_aura_size
            self.aura_program["u_point_size"].value = new_aura_size

        self.ctx.clear(0.015, 0.015, 0.03, 1.0)

        if self._n_reactive > 0:
            self.aura_vao.render(moderngl.POINTS, vertices=self._n_reactive)
        if self._bond_vertex_count > 0:
            self.bond_vao.render(moderngl.LINES, vertices=self._bond_vertex_count)
        if self._n_visible > 0:
            self.atlas_tex.use(location=0)
            self.letter_program["u_atlas"].value = 0
            self.letter_vao.render(
                moderngl.TRIANGLE_STRIP, vertices=4, instances=self._n_visible
            )
        # Bond-strength labels (small text on top of pair-bond lines).
        if self._label_count > 0:
            self.atlas_tex.use(location=0)
            self.letter_program["u_atlas"].value = 0
            self.letter_program["u_glyph_size"].value = self.LABEL_GLYPH_SIZE
            self.label_vao.render(
                moderngl.TRIANGLE_STRIP, vertices=4, instances=self._label_count
            )
            self.letter_program["u_glyph_size"].value = self.GLYPH_WORLD_SIZE

        # Pipe pass — opaque, so it occludes anything behind it.
        self.ctx.disable(moderngl.BLEND)
        self.pipe_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        self.ctx.enable(moderngl.BLEND)

        # PCA hover popup (top of stack, alpha-blended).
        if getattr(self, "_popup_enabled", False) and self._hover_token >= 0:
            if self._hover_token != self._hover_token_cached:
                self._composite_popup(self._hover_token)
                self._hover_token_cached = self._hover_token
            self._render_popup()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def on_key_event(self, key, action, modifiers) -> None:
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS and key in (keys.S, keys.SPACE):
            self._show_strain = not self._show_strain
            self.bond_program["u_show_strain"].value = 1 if self._show_strain else 0

    def on_mouse_press_event(self, x, y, button) -> None:
        wx, wy = self.screen_to_world(x, y)
        s = self.world.state
        keys = self.wnd.mouse

        if button == keys.right:
            self._is_panning = True
            return

        # Left button — try to grab a word, otherwise fall back to pan.
        if button == keys.left:
            picked = self._pick_letter(wx, wy)
            if picked is not None:
                self._begin_drag(picked, wx, wy)
            else:
                self._is_panning = True

    def on_mouse_release_event(self, x, y, button) -> None:
        keys = self.wnd.mouse
        if button == keys.right:
            self._is_panning = False
        if button == keys.left:
            self._is_panning = False
            self._end_drag()

    def on_mouse_drag_event(self, x, y, dx, dy) -> None:
        self._update_hover(x, y)
        if self._drag_letters.size > 0:
            wx, wy = self.screen_to_world(x, y)
            self._cursor_world = (wx, wy)
            return
        if self._is_panning:
            win_w, win_h = self.wnd.size
            l, r, b, t = self._view_rect()
            self._cam_x -= dx * (r - l) / win_w
            self._cam_y += dy * (t - b) / win_h   # screen-y flipped
            self._proj_dirty = True

    def on_mouse_position_event(self, x, y, dx, dy) -> None:
        # Fired when no button is held — used to drive the hover popup.
        self._update_hover(x, y)

    def on_mouse_scroll_event(self, x_offset, y_offset) -> None:
        # y_offset > 0 zooms in, < 0 zooms out.
        factor = 1.1 ** y_offset
        self._cam_zoom = max(0.2, min(10.0, self._cam_zoom * factor))
        self._proj_dirty = True

    # ------------------------------------------------------------------
    # Drag mechanics
    # ------------------------------------------------------------------

    def _pick_letter(self, wx: float, wy: float) -> int | None:
        """Return the visible letter index nearest the cursor within PICK_RADIUS,
        or None if no letter is close enough."""
        if self._n_visible == 0:
            return None
        sp = self._smoothed_pos[self._visible_idx]
        dx = sp[:, 0] - wx
        dy = sp[:, 1] - wy
        d2 = dx * dx + dy * dy
        k = int(np.argmin(d2))
        if d2[k] > self.PICK_RADIUS * self.PICK_RADIUS:
            return None
        return int(self._visible_idx[k])

    def _begin_drag(self, letter_index: int, cursor_x: float, cursor_y: float) -> None:
        s = self.world.state
        tid = int(s.token_id[letter_index])
        ci = int(s.chain_id[letter_index])
        # All letters belonging to the same token + same chain.
        same_token = (s.token_id == tid) & (s.chain_id == ci)
        idx = np.where(same_token)[0].astype(np.int32)
        if idx.size == 0:
            return
        cursor = np.array([cursor_x, cursor_y], dtype=np.float64)
        offsets = s.pos[idx] - cursor                # initial relative geometry
        self._drag_letters = idx
        self._drag_offsets = offsets
        self._cursor_world = (cursor_x, cursor_y)
        s.drag_indices = idx
        s.drag_targets = s.pos[idx].copy()           # start at current pos
        s.vel[idx] = 0.0                              # kill thermal velocity in grabbed beads

    def _end_drag(self) -> None:
        if self._drag_letters.size == 0:
            return
        s = self.world.state
        s.drag_indices = np.zeros(0, dtype=np.int32)
        s.drag_targets = np.zeros((0, 2), dtype=np.float64)
        self._drag_letters = np.zeros(0, dtype=np.int32)
        self._drag_offsets = np.zeros((0, 2), dtype=np.float64)

    def _update_drag_targets(self) -> None:
        if self._drag_letters.size == 0:
            return
        s = self.world.state
        cursor = np.array(self._cursor_world, dtype=np.float64)
        naive = cursor + self._drag_offsets
        # Clamp the per-frame motion of each drag target so a fast cursor
        # can't teleport letters and yank the chain through itself. The
        # letter then "walks" toward the cursor at MAX_DRAG_STEP per frame.
        actual = s.pos[self._drag_letters]
        delta = naive - actual
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        MAX_DRAG_STEP = 1.5  # world units per render frame
        factor = np.minimum(1.0, MAX_DRAG_STEP / np.maximum(dist, 1e-9))
        s.drag_targets = actual + delta * factor

    # ------------------------------------------------------------------
    # Geometry uploads
    # ------------------------------------------------------------------

    def _update_smoothed_positions(self) -> None:
        actual = self.world.state.pos.astype("f4")
        a = self.POSITION_SMOOTHING
        self._smoothed_pos = a * actual + (1.0 - a) * self._smoothed_pos

    def _compute_visible_tangents(self) -> np.ndarray:
        sp = self._smoothed_pos
        m = self._visible_idx.size
        prev_v = self._prev_neighbor
        next_v = self._next_neighbor
        tangent = np.zeros((m, 2), dtype=np.float32)
        tangent[:, 0] = 1.0
        prev_ok = prev_v >= 0
        next_ok = next_v >= 0
        both = prev_ok & next_ok
        only_next = ~prev_ok & next_ok
        only_prev = prev_ok & ~next_ok
        if both.any():
            tangent[both] = sp[next_v[both]] - sp[prev_v[both]]
        if only_next.any():
            tangent[only_next] = sp[next_v[only_next]] - sp[self._visible_idx[only_next]]
        if only_prev.any():
            tangent[only_prev] = sp[self._visible_idx[only_prev]] - sp[prev_v[only_prev]]
        norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        bad = norm[:, 0] < 1e-9
        tangent = tangent / np.clip(norm, 1e-9, None)
        if bad.any():
            tangent[bad] = (1.0, 0.0)
        return tangent

    def _upload_geometry(self) -> None:
        s = self.world.state
        sp = self._smoothed_pos

        pos_v = sp[self._visible_idx]
        self.letter_pos_buf.write(pos_v.tobytes())
        tangents = self._compute_visible_tangents()
        self.letter_tangent_buf.write(tangents.tobytes())

        if self._n_reactive > 0:
            self.aura_pos_buf.write(sp[self._reactive_idx].tobytes())

        backbone = s.backbone
        bp = s.base_pairs
        n_back = len(backbone)
        n_bp = len(bp)
        if n_back + n_bp == 0:
            self._bond_vertex_count = 0
            self._label_count = 0
            return

        if n_bp > 0:
            pairs = np.concatenate([backbone, bp], axis=0)
            kinds = np.concatenate(
                [np.zeros(n_back, dtype=np.int32), np.ones(n_bp, dtype=np.int32)]
            )
        else:
            pairs = backbone
            kinds = np.zeros(n_back, dtype=np.int32)

        verts = sp[pairs].reshape(-1, 2)
        kinds_per_vertex = np.repeat(kinds, 2).astype("i4")
        strain_per_vertex = s.strain[pairs].reshape(-1).astype("f4")

        self.bond_pos_buf.write(verts.tobytes())
        self.bond_kind_buf.write(kinds_per_vertex.tobytes())
        self.bond_strain_buf.write(strain_per_vertex.tobytes())
        self._bond_vertex_count = verts.shape[0]

        self._upload_bond_labels()

    def _upload_bond_labels(self) -> None:
        """Build per-character instances for the affinity numbers above each
        pair-bond and write them to the label_* buffers."""
        s = self.world.state
        bp = s.base_pairs
        if len(bp) == 0:
            self._label_count = 0
            return

        sp = self._smoothed_pos
        ti = s.token_id[bp[:, 0]]
        tj = s.token_id[bp[:, 1]]
        affinities = s.token_affinity[ti, tj]

        pi = sp[bp[:, 0]]
        pj = sp[bp[:, 1]]
        diff = pj - pi
        L = np.linalg.norm(diff, axis=1, keepdims=True)
        ok = L[:, 0] > 1e-6
        if not ok.any():
            self._label_count = 0
            return
        tangent = np.where(ok[:, None], diff / np.where(L > 1e-9, L, 1.0),
                           np.array([1.0, 0.0])).astype(np.float32)
        # Normal = +90° CCW rotation of tangent.
        normal = np.zeros_like(tangent)
        normal[:, 0] = -tangent[:, 1]
        normal[:, 1] = +tangent[:, 0]
        midpoint = ((pi + pj) * 0.5 + normal * self.LABEL_OFFSET_FROM_BOND).astype(np.float32)

        # Format texts and lay each character along the bond.
        texts = [f"{a:.2f}" for a in affinities]
        text_lens = np.array([len(t) for t in texts], dtype=np.int32)
        n_total = int(text_lens.sum())
        if n_total == 0 or n_total > self.MAX_LABEL_CHARS:
            self._label_count = min(n_total, self.MAX_LABEL_CHARS)
            if n_total > self.MAX_LABEL_CHARS:
                # Truncate by dropping the tail (rare).
                texts = texts[: self.MAX_LABEL_CHARS // 4]
                text_lens = np.array([len(t) for t in texts], dtype=np.int32)
                n_total = int(text_lens.sum())

        bond_idx = np.repeat(np.arange(len(texts)), text_lens)
        char_idx = np.concatenate([np.arange(L) for L in text_lens])
        size = self.LABEL_GLYPH_SIZE
        offsets = (char_idx + 0.5) * size - text_lens[bond_idx] * size * 0.5
        char_pos = midpoint[bond_idx] + tangent[bond_idx] * offsets[:, None]

        # UV per character.
        all_chars = "".join(texts)
        uv_min_arr = np.zeros((n_total, 2), dtype=np.float32)
        uv_max_arr = np.zeros((n_total, 2), dtype=np.float32)
        for ci, ch in enumerate(all_chars):
            uvmn, uvmx = self.atlas.uv_for(ch)
            uv_min_arr[ci] = uvmn
            uv_max_arr[ci] = uvmx
        tan_arr = tangent[bond_idx].astype(np.float32)
        col_arr = np.tile([1.0, 0.92, 0.55], (n_total, 1)).astype(np.float32)

        self.label_pos_buf.write(char_pos.astype(np.float32).tobytes())
        self.label_tangent_buf.write(tan_arr.tobytes())
        self.label_uv_min_buf.write(uv_min_arr.tobytes())
        self.label_uv_max_buf.write(uv_max_arr.tobytes())
        self.label_color_buf.write(col_arr.tobytes())
        self._label_count = n_total


def run(world: World, font_path: str | None = None) -> None:
    HamletApp._WORLD = world
    HamletApp._FONT_PATH = font_path
    mglw.run_window_config(HamletApp)
