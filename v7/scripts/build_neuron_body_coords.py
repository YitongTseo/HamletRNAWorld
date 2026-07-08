"""Project the OpenWorm 3D neuron positions onto a canonical 2D worm body
plan for the x-ray visualization. Run once; output is checked in.

Mapping:
  anatomical_y → axial      [0, 1]   head=0, tail=1
  anatomical_x → lateral   [-1, +1]  worm-left negative, worm-right positive
  anatomical_z → dv        [-1, +1]  ventral negative, dorsal positive
                                    (kept for future side-view rendering)

We deliberately skip PCA here. SVD of the 301-neuron position cloud shows
PC1 ≈ y with alignment 0.997 — the data is already canonically axis-aligned
in OpenWorm's coords, and using PCA-derived axes would just rotate the
result by a few degrees with no payoff.

Output: cache/neuron_body_coords.json
    {
      "n_neurons": 301,
      "axial_extent": <world-coord length pre-normalization>,
      "lateral_extent": <world-coord width pre-normalization>,
      "neurons": { "ASEL": {"axial": ..., "lateral": ..., "dv": ...}, ... }
    }
"""
from __future__ import annotations

import json
from pathlib import Path

V6_ROOT = Path(__file__).resolve().parent.parent
SRC = V6_ROOT / "sim" / "neuron_positions.json"
OUT = V6_ROOT / "cache" / "neuron_body_coords.json"


def main():
    positions = json.loads(SRC.read_text())
    names = list(positions.keys())

    ys = [positions[n][1] for n in names]
    xs = [positions[n][0] for n in names]
    zs = [positions[n][2] for n in names]

    y_min, y_max = min(ys), max(ys)
    # Symmetric around 0 for lateral and dv so left/right and dorsal/ventral
    # stay signed. abs() of the larger half gives the half-extent.
    x_half = max(abs(min(xs)), abs(max(xs)), 1e-9)
    z_half = max(abs(min(zs)), abs(max(zs)), 1e-9)
    y_extent = y_max - y_min

    neurons = {}
    for n in names:
        x, y, z = positions[n]
        neurons[n] = {
            "axial":   round((y - y_min) / y_extent, 4),
            "lateral": round(x / x_half, 4),
            "dv":      round(z / z_half, 4),
        }

    payload = {
        "n_neurons": len(names),
        "y_min": y_min, "y_max": y_max,
        "axial_extent": y_extent,
        "lateral_half_extent": x_half,
        "dv_half_extent": z_half,
        "neurons": neurons,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload))
    print(f"Saved {OUT} ({OUT.stat().st_size // 1024} KB)")
    print(f"  axial extent in OpenWorm units: {y_extent:.1f}")
    print(f"  lateral half-extent: {x_half:.1f}")
    print(f"  dv half-extent: {z_half:.1f}")
    print(f"  head-most neurons: {sorted([(n, neurons[n]['axial']) for n in names], key=lambda x: x[1])[:5]}")
    print(f"  tail-most neurons: {sorted([(n, neurons[n]['axial']) for n in names], key=lambda x: -x[1])[:5]}")


if __name__ == "__main__":
    main()
