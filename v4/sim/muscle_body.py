"""Muscle-driven worm body.

EXPERIMENTAL — not the default body. The IK chain in `worm.py` is currently
the default because this version doesn't produce nice locomotion waves with
the GoPiGo connectome we ported. The stretch-receptive transport works in
principle, but the connectome lacks rhythmic head-ganglion neurons (RIA,
SMD, RMD) that would seed an oscillation for the wave to propagate. Without
that oscillation, the body just drifts between static curved poses every
brain tick.

To revisit this productively you'd need ONE of:
  - An explicit head oscillator labelled as a model of RIA/SMD (biologically
    motivated, but conceptually a separate addition).
  - Stretch-receptive synapses added to `weights.json` so B-class motor
    neurons drive themselves via curvature feedback (closest to real biology;
    Wen et al. 2012). Real research project.
  - Swap in a more faithful connectome model (c302 / NeuroML).

Replaces the kinematic IK-chain body with a midline integrated from
per-segment dorsal-ventral muscle differentials read from the connectome.

Anatomy:
- 17 body-wall muscle segments along the trunk (indices 07..23 in worm-sim's
  notation), with 4 quadrants each: dorsal-left/right, ventral-left/right.
- Real C. elegans crawls by alternating dorsal and ventral contractions in a
  travelling wave; we drive each segment's local bend angle from
  (D_left + D_right) - (V_left + V_right) — the dorsal-ventral differential.
- The connectome's L-R sum still drives global heading, mirroring the
  GoPiGo/worm-sim approach for steering. Speed is the total muscle activity
  magnitude.

Smoothing:
- Connectome ticks at 2 Hz and zeros muscle outputs after each tick. We
  exponentially-smooth a `force` value per quadrant per segment at the body
  rate (60 Hz) so the midline can update smoothly between brain ticks.

Midline reconstruction:
- Walk from head position, in the trailing direction (-facing), rotating
  by `bend[i]` at each joint.
- Catmull-Rom-interpolate the 18 anchor points up to 200 samples for the
  viewer (its TubeGeometry-replacement renders linear segments, so we
  pre-smooth here).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# Mapping from segment index 0..16 → muscle name suffix "07".."23".
MUSCLE_INDICES = list(range(7, 24))
N_MUSCLE_SEG = len(MUSCLE_INDICES)
# Render-time interpolation density.
N_RENDER_SEG = 200


def _catmull_rom_interp(anchors: np.ndarray, n_out: int) -> np.ndarray:
    """Sample a Catmull-Rom spline through `anchors` at `n_out` evenly spaced
    points. anchors shape: (k, 2). Returns shape (n_out, 2)."""
    k = anchors.shape[0]
    if k < 2:
        return np.repeat(anchors, n_out, axis=0)
    # Pad endpoints so the curve passes through the first and last anchors.
    pad_pre = anchors[0] + (anchors[0] - anchors[1])
    pad_post = anchors[-1] + (anchors[-1] - anchors[-2])
    pts = np.vstack([pad_pre, anchors, pad_post])  # (k+2, 2)

    # n_out samples spread across (k-1) Catmull-Rom segments.
    t = np.linspace(0, k - 1, n_out)  # global parameter
    seg_idx = np.clip(t.astype(int), 0, k - 2)
    local_t = (t - seg_idx).reshape(-1, 1)

    p0 = pts[seg_idx]      # = anchors[seg_idx - 1] in unpadded indexing
    p1 = pts[seg_idx + 1]  # = anchors[seg_idx]
    p2 = pts[seg_idx + 2]  # = anchors[seg_idx + 1]
    p3 = pts[seg_idx + 3]  # = anchors[seg_idx + 2]

    t2 = local_t * local_t
    t3 = t2 * local_t
    out = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * local_t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )
    return out


@dataclass
class MuscleBody:
    origin_x: float = 0.0
    origin_y: float = 0.0
    body_length: float = 800.0
    facing_dir: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    speed: float = 0.0
    # Internal clock used to drive the central pattern generator (see below).
    _t: float = 0.0
    body_dt: float = 1.0 / 60.0

    # Per-segment smoothed force, one array per quadrant.
    force_DL: np.ndarray = field(init=False)
    force_DR: np.ndarray = field(init=False)
    force_VL: np.ndarray = field(init=False)
    force_VR: np.ndarray = field(init=False)
    bend: np.ndarray = field(init=False)

    # Smoothing time constants.
    force_alpha: float = 0.10        # per body tick (60 Hz) → ≈ 100 ms time const
    bend_scale: float = 0.018        # rad per unit muscle differential
    speed_scale: float = 0.012
    speed_alpha: float = 0.08
    steer_scale: float = 0.0006      # rad per (L-R) per body tick

    # ── Stretch-receptive proprioceptive coupling ──────────────────────────
    # Real C. elegans B-class motor neurons (DB1–7, VB1–11) have a long
    # posterior process that mechanosenses curvature in the body region
    # ANTERIOR to where they innervate; this closes a fast head→tail loop
    # that produces the locomotion wave (Wen et al. 2012). The simplified
    # GoPiGo connectome we ported has no such sensors in its weight table.
    # We approximate the loop here at the body level: each segment's bend
    # passively converges toward its anterior neighbor's bend, so when the
    # head bend changes (driven by the connectome's L-R asymmetry, which is
    # what command interneurons → head ganglion oscillators do in vivo),
    # the deflection propagates posteriorly.
    stretch_alpha: float = 0.22      # per-tick rate of "i takes on (i-1)"
    stretch_damping: float = 0.012   # passive relaxation toward straight
    head_alpha: float = 0.18         # how fast head bend follows connectome

    # CPG — kept in the dataclass so the field still exists, but disabled
    # by default. The wave should now come from the stretch loop instead.
    cpg_freq_hz: float = 0.45
    cpg_amp: float = 0.0
    cpg_wavelength_seg: float = 13.0
    cpg_activation_floor: float = 35.0

    def __post_init__(self):
        self.target_x = self.origin_x
        self.target_y = self.origin_y
        self.force_DL = np.zeros(N_MUSCLE_SEG)
        self.force_DR = np.zeros(N_MUSCLE_SEG)
        self.force_VL = np.zeros(N_MUSCLE_SEG)
        self.force_VR = np.zeros(N_MUSCLE_SEG)
        self.bend = np.zeros(N_MUSCLE_SEG)

    def consume(self, brain) -> None:
        """Pull the latest per-muscle activations from the brain. Called once
        per body tick (the brain itself only refreshes every BRAIN_PERIOD).

        Note: worm-sim's M_RIGHT list has a typo at index 21 (MDL21/MVL21
        instead of MDR21/MVR21) that we preserve for behavioral parity, so
        MDR21/MVR21 may be missing from the activations dict. We default
        those to 0 — equivalent to "not driven" in the original sim."""
        a = self.force_alpha
        ma = brain.muscle_activations
        for i, seg in enumerate(MUSCLE_INDICES):
            tag = f"{seg:02d}"
            self.force_DL[i] += a * (ma.get(f"MDL{tag}", 0.0) - self.force_DL[i])
            self.force_DR[i] += a * (ma.get(f"MDR{tag}", 0.0) - self.force_DR[i])
            self.force_VL[i] += a * (ma.get(f"MVL{tag}", 0.0) - self.force_VL[i])
            self.force_VR[i] += a * (ma.get(f"MVR{tag}", 0.0) - self.force_VR[i])

        # Connectome-driven head drive. In real worms the head bends rhythmically
        # because of head-ganglion command/oscillator neurons (RIA, SMDs, etc.).
        # Here we use the body's L-R muscle asymmetry as a proxy — when the
        # connectome's left-side muscles are firing harder than the right, the
        # head bends one way; vice versa for the other.
        L = float(self.force_DL.sum() + self.force_VL.sum())
        R = float(self.force_DR.sum() + self.force_VR.sum())
        # Stronger head drive so the connectome's small L-R differential
        # actually produces visible head bend that the stretch loop can
        # propagate. (This is the experimental knob — too high and the head
        # over-flexes; too low and the body stays straight.)
        head_target = self.bend_scale * (L - R) * 6.0

        # Per-segment connectome contribution (small — provides a baseline
        # spatially-distributed drive but doesn't generate rhythm by itself).
        D = self.force_DL + self.force_DR
        V = self.force_VL + self.force_VR
        connectome_drive = 0.25 * self.bend_scale * (D - V)

        # ── Stretch-receptive transport ────────────────────────────────────
        # Each segment converges toward its ANTERIOR neighbor's bend. This is
        # the head→tail propagator. With small stretch_alpha the wave moves
        # slowly enough to look like crawling rather than slithering.
        new_bend = self.bend.copy()
        # Head segment: driven by connectome's L-R asymmetry.
        new_bend[0] += self.head_alpha * (head_target - new_bend[0])
        # Posterior segments: lerp toward anterior neighbor.
        new_bend[1:] += self.stretch_alpha * (self.bend[:-1] - self.bend[1:])
        # Distributed connectome drive (small contribution everywhere).
        new_bend += 0.03 * (connectome_drive - new_bend)
        # Passive relaxation so a static deflection slowly returns to 0.
        new_bend *= (1.0 - self.stretch_damping)

        # Optional CPG layer (off by default — kept for ablation).
        if self.cpg_amp > 0:
            omega = 2.0 * math.pi * self.cpg_freq_hz
            k = 2.0 * math.pi / self.cpg_wavelength_seg
            seg_idx = np.arange(N_MUSCLE_SEG, dtype=float)
            activity = float(np.abs(self.force_DL).sum() + np.abs(self.force_DR).sum()
                             + np.abs(self.force_VL).sum() + np.abs(self.force_VR).sum())
            gate = min(1.0, activity / max(self.cpg_activation_floor, 1e-6))
            new_bend += gate * self.cpg_amp * np.sin(omega * self._t - k * seg_idx)

        self.bend = new_bend

        # Global heading + speed.
        target_speed = self.speed_scale * (L + R)
        self.speed += self.speed_alpha * (target_speed - self.speed)
        self.facing_dir += self.steer_scale * (L - R)

    def step(self) -> None:
        """One body tick (intended at 60 Hz). Advance the head, integrate
        the midline from the current bend angles."""
        self._t += self.body_dt
        self.target_x += math.cos(self.facing_dir) * self.speed
        self.target_y -= math.sin(self.facing_dir) * self.speed  # y-down

    def midline(self) -> list[tuple[float, float]]:
        """Return the smoothed midline polyline (head → tail), 200 points."""
        seg_len = self.body_length / N_MUSCLE_SEG
        # Walk from head in the trailing (-facing) direction, applying the
        # per-segment bend at each joint.
        x, y = self.target_x, self.target_y
        angle = self.facing_dir + math.pi
        anchors = np.empty((N_MUSCLE_SEG + 1, 2))
        anchors[0] = (x, y)
        for i in range(N_MUSCLE_SEG):
            angle += self.bend[i]
            x += math.cos(angle) * seg_len
            y -= math.sin(angle) * seg_len  # y-down
            anchors[i + 1] = (x, y)
        smooth = _catmull_rom_interp(anchors, N_RENDER_SEG + 1)
        return [(float(p[0]), float(p[1])) for p in smooth]
