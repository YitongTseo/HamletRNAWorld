"""Headless sim sanity check — no server, no browser. Run a few seconds of
world ticks and print summary stats. Useful for catching connectome /
body bugs without spinning up the viewer."""
from __future__ import annotations

import time

from sim.world import World


def main():
    w = World()
    t0 = time.monotonic()
    n_ticks = 60 * 5  # 5 seconds worth of body ticks at 60 Hz
    for i in range(n_ticks):
        # Simulate real time so the brain ticks the right number of times.
        now = t0 + i * (1 / 60)
        w.step(now)
    snap = w.snapshot()
    print(f"midline points: {len(snap['midline'])}")
    print(f"head: {snap['head']}  facing={snap['facing']}  speed={snap['speed']}")
    print(f"motor L={snap['motor']['L']}  R={snap['motor']['R']}")
    print(f"stim: {snap['stim']}")
    # Spot-check that the worm actually moved.
    head_dx = abs(snap['head'][0] - 1600/2) + abs(snap['head'][1] - 1000/2)
    print(f"head displacement from start: {head_dx:.2f} px")
    if head_dx < 1.0:
        print("WARNING: worm barely moved — possible motor wiring issue")


if __name__ == "__main__":
    main()
