"""Experiment 1 — pool construction from historical worm data.

A "pool" is one flask's generation: 16 worms, each represented by its
historically-judged 15-token windows (from scores.jsonl — which conveniently
carries both the token text we re-judge and a historical (emotional, coherence)
score we use only to *stratify* window sampling). No simulation is rerun.

See docs/superpowers/specs/2026-07-17-judge-noise-and-sigma-control-experiments.md
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

V7 = Path(__file__).resolve().parent.parent.parent  # .../v7

# The 4 pools: (proc, flask, gen). Gen-28 = deep in the sigma-runaway regime.
POOLS: list[tuple[str, str, int]] = [
    ("poetry-1", "flask_1", 28),
    ("poetry-1", "flask_2", 28),
    ("poetry-2", "flask_1", 28),
    ("poetry-2", "flask_2", 28),
]


@dataclass(frozen=True)
class Window:
    idx: int
    tokens: tuple[str, ...]
    hist_e: int  # historical emotional score (for stratification only)
    hist_c: int  # historical coherence score (for stratification only)

    @property
    def hist_q(self) -> float:
        """Per-window quality in the engine's fitness shape (GAMMA=1.5,
        emotional weight 1.5) — used to stratify, matching what fitness rewards."""
        return 1.5 * (self.hist_e / 100.0) ** 1.5 + (self.hist_c / 100.0) ** 1.5


def pool_id(proc: str, flask: str, gen: int) -> str:
    return f"{proc}.{flask}.gen-{gen:04d}"


def load_pool(proc: str, flask: str, gen: int) -> dict[str, list[Window]]:
    """worm_name -> list[Window] for every worm in the generation."""
    gen_dir = V7 / "data" / proc / "generations" / flask / f"gen-{gen:04d}"
    worms: dict[str, list[Window]] = {}
    for wd in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
        sj = wd / "scores.jsonl"
        if not sj.exists():
            continue
        wins: list[Window] = []
        for line in sj.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            wins.append(Window(
                idx=int(d["idx"]),
                tokens=tuple(d["tokens"]),
                hist_e=int(d["emotional"]),
                hist_c=int(d["coherence"]),
            ))
        if wins:
            worms[wd.name] = wins
    return worms


# --- window-sampling strategies (Lever 1: how a worm is represented) ---------

def sample_random(windows: list[Window], m: int, rng: random.Random) -> list[Window]:
    """m uniformly-random windows (order restored by idx for stable prompts)."""
    if m >= len(windows):
        return sorted(windows, key=lambda w: w.idx)
    picked = rng.sample(windows, m)
    return sorted(picked, key=lambda w: w.idx)


def sample_stratified(windows: list[Window], m: int, rng: random.Random) -> list[Window]:
    """Sort by historical quality, split into m contiguous bins, pick one random
    window per bin — guaranteeing the sample spans the worm's low->high range
    rather than possibly clustering all-mediocre."""
    if m >= len(windows):
        return sorted(windows, key=lambda w: w.idx)
    ordered = sorted(windows, key=lambda w: w.hist_q)
    n = len(ordered)
    picked: list[Window] = []
    for b in range(m):
        lo = (b * n) // m
        hi = ((b + 1) * n) // m
        if hi <= lo:
            hi = lo + 1
        picked.append(ordered[rng.randrange(lo, min(hi, n))])
    return sorted(picked, key=lambda w: w.idx)


SAMPLERS = {"random": sample_random, "stratified": sample_stratified}


def represent(windows: list[Window], sampling: str, m: int, rng: random.Random) -> list[Window]:
    return SAMPLERS[sampling](windows, m, rng)
