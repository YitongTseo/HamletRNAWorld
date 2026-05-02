"""Headless smoke test for the simulation core (no window).

Runs the sim for a fixed number of steps and reports bond counts and a few
position stats. Useful to verify the physics works before debugging the GL.
"""
from __future__ import annotations

import time

import numpy as np

from sim.world import World


def main():
    w = World()
    s = w.state
    print(f"sequence: N={s.n}, bases={''.join('AUGC'[b] for b in s.bases)}")
    print(f"initial x range: [{s.pos[:,0].min():.2f}, {s.pos[:,0].max():.2f}]")
    print(f"initial y range: [{s.pos[:,1].min():.2f}, {s.pos[:,1].max():.2f}]")

    total_steps = 80_000
    log_every = 5_000
    t0 = time.time()
    for step in range(0, total_steps, log_every):
        w.step(log_every)
        elapsed = time.time() - t0
        x_lead = s.pos[0, 0]
        x_tail = s.pos[-1, 0]
        n_bp = len(s.base_pairs)
        max_speed = float(np.linalg.norm(s.vel, axis=1).max())
        print(
            f"step {s.step_count:6d} | bp={n_bp:3d} | lead.x={x_lead:+.2f} "
            f"tail.x={x_tail:+.2f} | vmax={max_speed:.2f} | t={elapsed:.2f}s"
        )

    # Final bond inventory: should look like an antiparallel paired stem.
    print("\nFinal base pairs (i, j):")
    if len(s.base_pairs) == 0:
        print("  (none)")
    else:
        pairs_sorted = s.base_pairs[np.argsort(s.base_pairs[:, 0])]
        for i, j in pairs_sorted:
            print(f"  {int(i):3d} -- {int(j):3d}   ({'AUGC'[s.bases[i]]}-{'AUGC'[s.bases[j]]})")


if __name__ == "__main__":
    main()
