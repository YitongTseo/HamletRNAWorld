"""Determinism test: given the same seed, two World instances must produce
identical eaten-word sequences and identical head positions.

This is the load-bearing property for v6's persistence model — without it,
saving a worm's seed doesn't recreate its behavior."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sim.world import World


def _run(seed: int, n_ticks: int) -> tuple[list[tuple[int, int, str]], list[tuple[float, float]]]:
    w = World(seed=seed)
    eaten: list[tuple[int, int, str]] = []
    head_trace: list[tuple[float, float]] = []
    for _ in range(n_ticks):
        w.tick()
        eaten.extend(w.drain_eaten_words())
        head_trace.append((w.worm.target_x, w.worm.target_y))
    return eaten, head_trace


def test_same_seed_same_trajectory():
    eaten_a, head_a = _run(seed=42, n_ticks=600)  # 10 s of sim
    eaten_b, head_b = _run(seed=42, n_ticks=600)
    assert eaten_a == eaten_b, "Eaten-word sequence diverged for same seed"
    assert head_a == head_b, "Head position trace diverged for same seed"


def test_different_seeds_different_trajectories():
    eaten_a, head_a = _run(seed=1, n_ticks=600)
    eaten_b, head_b = _run(seed=2, n_ticks=600)
    # Heads should differ at SOME tick; if not, the seed isn't influencing
    # anything that matters.
    assert head_a != head_b, "Different seeds produced identical head trace"


if __name__ == "__main__":
    test_same_seed_same_trajectory()
    print("PASS: same seed → same trajectory")
    test_different_seeds_different_trajectories()
    print("PASS: different seeds → different trajectories")
