"""moderngl-window app: steps the sim and renders glowing beads + bonds."""
from __future__ import annotations

from pathlib import Path

import moderngl
import moderngl_window as mglw
import numpy as np

from sim.world import World

SHADER_DIR = Path(__file__).parent / "shaders"


def _ortho(left: float, right: float, bottom: float, top: float,
           near: float = -1.0, far: float = 1.0) -> np.ndarray:
    """Column-major 4x4 orthographic projection, ready to upload via .write()."""
    return np.array([
        2.0 / (right - left), 0.0, 0.0, 0.0,                         # col 0
        0.0, 2.0 / (top - bottom), 0.0, 0.0,                         # col 1
        0.0, 0.0, -2.0 / (far - near), 0.0,                          # col 2
        -(right + left) / (right - left),
        -(top + bottom) / (top - bottom),
        -(far + near) / (far - near),
        1.0,                                                         # col 3
    ], dtype="f4")


class RNAApp(mglw.WindowConfig):
    title = "RNA World v2"
    gl_version = (3, 3)
    window_size = (1600, 900)
    aspect_ratio = 16 / 9
    resizable = True
    samples = 4  # MSAA — smooths bond lines and bead edges

    # Visible world rectangle (matches aspect_ratio).
    WORLD_X = (-16.0, 16.0)
    WORLD_Y = (-9.0, 9.0)
    BEAD_PIXEL_SIZE = 28.0
    SIM_STEPS_PER_FRAME = 8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.world = World()
        n = self.world.state.n
        self._show_strain = False

        # --- Shader programs ---
        self.bead_program = self.ctx.program(
            vertex_shader=(SHADER_DIR / "bead.vert").read_text(),
            fragment_shader=(SHADER_DIR / "bead.frag").read_text(),
        )
        self.bond_program = self.ctx.program(
            vertex_shader=(SHADER_DIR / "bond.vert").read_text(),
            fragment_shader=(SHADER_DIR / "bond.frag").read_text(),
        )

        proj = _ortho(*self.WORLD_X, *self.WORLD_Y)
        self.bead_program["u_proj"].write(proj.tobytes())
        self.bond_program["u_proj"].write(proj.tobytes())
        self.bead_program["u_point_size"].value = self.BEAD_PIXEL_SIZE
        self.bond_program["u_intensity"].value = 0.7
        self.bond_program["u_show_strain"].value = 0

        # --- Bead buffers ---
        self.bead_pos_buf = self.ctx.buffer(reserve=n * 2 * 4, dynamic=True)
        self.bead_base_buf = self.ctx.buffer(
            self.world.state.bases.astype("i4").tobytes()
        )
        self.bead_vao = self.ctx.vertex_array(
            self.bead_program,
            [
                (self.bead_pos_buf, "2f", "in_pos"),
                (self.bead_base_buf, "1i", "in_base"),
            ],
        )

        # --- Bond buffers --- (size for backbone + worst-case base pairs).
        # Each segment contributes 2 vertices; per vertex we store position,
        # kind (backbone/bp), and strain.
        max_segments = (n - 1) + (n // 2) + 4
        self.bond_pos_buf = self.ctx.buffer(
            reserve=max_segments * 2 * 2 * 4, dynamic=True
        )
        self.bond_kind_buf = self.ctx.buffer(
            reserve=max_segments * 2 * 4, dynamic=True
        )
        self.bond_strain_buf = self.ctx.buffer(
            reserve=max_segments * 2 * 4, dynamic=True
        )
        self.bond_vao = self.ctx.vertex_array(
            self.bond_program,
            [
                (self.bond_pos_buf, "2f", "in_pos"),
                (self.bond_kind_buf, "1i", "in_kind"),
                (self.bond_strain_buf, "1f", "in_strain"),
            ],
        )
        self._bond_vertex_count = 0

        # --- GL state ---
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE  # additive
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        try:
            self.ctx.line_width = 2.0  # ignored on some drivers; harmless
        except Exception:
            pass

    # ------------------------------------------------------------------

    def on_render(self, time: float, frame_time: float) -> None:
        self.world.step(self.SIM_STEPS_PER_FRAME)
        self._upload_geometry()

        self.ctx.clear(0.015, 0.015, 0.03, 1.0)
        if self._bond_vertex_count > 0:
            self.bond_vao.render(moderngl.LINES, vertices=self._bond_vertex_count)
        self.bead_vao.render(moderngl.POINTS, vertices=self.world.state.n)

    def on_key_event(self, key, action, modifiers) -> None:
        # Toggle strain visualization on key press. Both 'S' and SPACE work.
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS and key in (keys.S, keys.SPACE):
            self._show_strain = not self._show_strain
            self.bond_program["u_show_strain"].value = 1 if self._show_strain else 0

    # ------------------------------------------------------------------

    def _upload_geometry(self) -> None:
        s = self.world.state
        self.bead_pos_buf.write(s.pos.astype("f4").tobytes())

        backbone = s.backbone
        bp = s.base_pairs
        n_back = len(backbone)
        n_bp = len(bp)

        if n_back + n_bp == 0:
            self._bond_vertex_count = 0
            return

        if n_bp > 0:
            pairs = np.concatenate([backbone, bp], axis=0)
            kinds = np.concatenate(
                [np.zeros(n_back, dtype=np.int32), np.ones(n_bp, dtype=np.int32)]
            )
        else:
            pairs = backbone
            kinds = np.zeros(n_back, dtype=np.int32)

        # Each pair contributes two vertices (one per endpoint).
        verts = s.pos[pairs].reshape(-1, 2).astype("f4")
        kinds_per_vertex = np.repeat(kinds, 2).astype("i4")
        # Strain is meaningful only on backbone-bond endpoints; for bp endpoints
        # we still pass s.strain (it will be hidden by the kind gate in shader).
        strain_per_vertex = s.strain[pairs].reshape(-1).astype("f4")

        self.bond_pos_buf.write(verts.tobytes())
        self.bond_kind_buf.write(kinds_per_vertex.tobytes())
        self.bond_strain_buf.write(strain_per_vertex.tobytes())
        self._bond_vertex_count = verts.shape[0]


def run() -> None:
    mglw.run_window_config(RNAApp)
